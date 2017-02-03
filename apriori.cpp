#include "iostream"
#include "fstream"
#include "vector"
#include "string"
#include "sstream"
#include "unordered_map"
#include "unordered_set"
#include "set"
#include "mutex"
#include "atomic"
#include "thread"

constexpr int TRANSACTION_POOL_SIZE = 100;

//using items_t = std::set<int>;
using items_t = std::vector<int>;
using itemset_t = std::unordered_map<items_t, int>;
using transaction_t = std::unordered_set<int>;
using transactions_t = std::vector<transaction_t>;

namespace std {
    template <>
    class hash<std::vector<int>> {
    public:
        // It comes from http://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
        std::size_t operator()(std::vector<int> const& vec) const {
            std::size_t seed = vec.size();
            for(auto& i : vec) {
                seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
};

transactions_t readData(const std::string& filename) {
    std::ifstream data(filename);
    std::string line;
    std::stringstream ss;
    std::vector<std::unordered_set<int>> transactions;
    while (std::getline(data, line)) {
        ss.clear();
        ss << line;
        std::unordered_set<int> items;
        int item;
        while (ss >> item) {
            items.insert(item);
        }
        transactions.push_back(std::move(items));
    }
    return transactions;
}

class AsyncTransactionFeeder {
public:
    std::string filename;
    std::fstream file;
    transactions_t transactionsPool;
    std::atomic<int> poolInPos;
    std::atomic<int> poolOutPos;
    std::atomic<bool> dataEnd;
    std::atomic<bool> fitInMem;

    AsyncTransactionFeeder() :
            transactionsPool(TRANSACTION_POOL_SIZE) {
        poolInPos = 0;
        poolOutPos = 0;
        dataEnd = false;
        fitInMem = true;
    }

    void attachFile(const std::string& filename) {
        this->filename = filename;
        file.open(filename);
    }

    void reset() {
        if (fitInMem) {
            poolOutPos = 0;
            dataEnd = true;
        }
        file.close();
        file.open(filename);
        poolInPos = poolOutPos = 0;
        dataEnd = false;
        transactionsPool.clear();
        transactionsPool.resize(TRANSACTION_POOL_SIZE);
    }

    void start() {
        if (dataEnd) {
            return;
        }
        std::thread reader(&AsyncTransactionFeeder::asyncRead, this);
        reader.detach();
    }

    void asyncRead() {
        std::string line;
        std::stringstream ss;
        while (std::getline(file, line)) {
            ss.clear();
            ss << line;
            while ((poolInPos + 1) % TRANSACTION_POOL_SIZE == poolOutPos) {}
            auto& transaction = transactionsPool[poolInPos];
            transaction.clear();
            int item;
            while (ss >> item) {
                transaction.insert(item);
            }
            if (poolInPos + 1 == TRANSACTION_POOL_SIZE) {
                fitInMem = false;
            }
            poolInPos = (poolInPos + 1) % TRANSACTION_POOL_SIZE;
        }
        dataEnd.store(true);
    }

    const transaction_t* getNextTransaction() {
        while (true) {
            if (poolInPos == poolOutPos) {
                if (dataEnd) {
                    return nullptr;
                }
                continue;
            }
            auto transaction = &transactionsPool[poolOutPos];
            poolOutPos = (poolOutPos + 1) % TRANSACTION_POOL_SIZE;
            return transaction;
        }
    }

};

AsyncTransactionFeeder feeder;

bool combinable(const items_t& items1, const items_t& items2) {
    auto it1 = items1.begin();
    auto it2 = items2.begin();
    auto it1End = std::prev(items1.end());
    while (it1 != it1End) {
        if (*it1 == *it2) {
            ++it1;
            ++it2;
            continue;
        } else {
            return false;
        }
    }
    return true;
}

bool prune(const items_t& items, const itemset_t& L) {
    for (auto it = items.begin(); it != items.end(); ++it) {
        auto sub(items);
        sub.erase(sub.begin() + (it - items.begin()));
        if (L.find(sub) == L.end()) {
            return true;
        }
    }
    return false;
}

itemset_t generateC(itemset_t& L) {
    itemset_t C;
    for (auto i = L.begin(); i != L.end(); ++i) {
        for (auto j = std::next(i); j != L.end(); ++j) {
            if (combinable(i->first, j->first)) {
                items_t items(i->first);
                items.push_back(*std::prev(j->first.end()));
                std::sort(items.begin(), items.end());
                if (!prune(items, L)) {
                    C[items] = 0;
                }
            }
        }
    }
    return C;
}

itemset_t L1(int minSupp) {
    std::unordered_map<int, int> count;
    feeder.reset();
    feeder.start();
    while (true) {
        auto pTransaction = feeder.getNextTransaction();
        if (!pTransaction) {
            break;
        }
        for (auto& item : *pTransaction) {
            if (count.find(item) != count.end()) {
                count[item] += 1;
            } else {
                count[item] = 1;
            }
        }
    }
    itemset_t L;
    for (auto it = count.begin(); it != count.end(); ++it) {
        if (it->second >= minSupp) {
            std::vector<int> items{it->first};
            L[items] = it->second;
        }
    }
    return L;
}

itemset_t generateL(itemset_t& C, int minSupp) {
    feeder.reset();
    feeder.start();
    while (true) {
        auto pTransaction = feeder.getNextTransaction();
        if (!pTransaction) {
            break;
        }
        for (auto it = C.begin(); it != C.end(); ++it) {
            bool in = true;
            for (auto j = it->first.begin(); j != it->first.end(); ++j) {
                if (pTransaction->find(*j) == pTransaction->end()) {
                    in = false;
                    break;
                }
            }
            if (in) {
                it->second += 1;
            }
        }
    }
    itemset_t L;
    for (auto it = C.begin(); it != C.end(); ++it) {
        if (it->second >= minSupp) {
            L.insert(*it);
        }
    }
    return L;
}

//void generateStrongRule(items_t)

int main() {
    int minSupport = 2;
//    std::string filename("/Users/Shangtong/GitHub/DataMining/cmake-build-debug/dataS.txt");
    std::string filename("/Users/Shangtong/GitHub/DataMining/cmake-build-debug/in.txt");
    feeder.attachFile(filename);
//    auto transactions = readData(filename);
    auto L = L1(minSupport);
    while (true) {
        std::cout << "Freq" << std::endl;
        for (auto it = L.begin(); it != L.end(); ++it) {
            for (auto& item : it->first) {
                std::cout << item << " ";
            }
            std::cout << "-> " << it->second << std::endl;
        }
        if (L.empty()) {
            break;
        }
        auto C = generateC(L);
        std::cout << "Candidate:" << std::endl;
        for (auto it = C.begin(); it != C.end(); ++it) {
            for (auto& item : it->first) {
                std::cout << item << " ";
            }
            std::cout << "-> " << it->second << std::endl;
        }

        L = generateL(C, minSupport);
    }
    return 0;
}