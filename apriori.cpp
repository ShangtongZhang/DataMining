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

constexpr int TRANSACTION_POOL_SIZE = 5;

//using items_t = std::set<int>;
using items_t = std::vector<int>;
using itemset_t = std::unordered_map<items_t, int>;
using transaction_t = std::unordered_set<int>;
using transactions_t = std::vector<transaction_t >;

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

    AsyncTransactionFeeder(const std::string& filename) :
            transactionsPool(TRANSACTION_POOL_SIZE),
            filename(filename) {
        file.open(filename);
        poolInPos.store(0);
        poolOutPos.store(0);
        dataEnd.store(false);
    }

    void start() {
        std::thread reader(&AsyncTransactionFeeder::asyncRead, this);
    }

    void asyncRead() {
        std::string line;
        std::stringstream ss;
        while (file >> line) {
            ss.clear();
            ss << line;
            while ((poolInPos + 1) % TRANSACTION_POOL_SIZE == poolOutPos) {}
            auto& transaction = transactionsPool[poolInPos];
            transaction.clear();
            int item;
            while (ss >> item) {
                transaction.insert(item);
            }
            poolInPos = (poolInPos + 1) % TRANSACTION_POOL_SIZE;
        }
        dataEnd.store(true);
    }

    const transaction_t* getNextTransaction() {
        while (true) {
            if (dataEnd) {
                return nullptr;
            }
            if (poolInPos == poolOutPos) {
                continue;
            }
            auto transaction = &transactionsPool[poolOutPos];
            poolOutPos = (poolOutPos + 1) % TRANSACTION_POOL_SIZE;
            return transaction;
        }
    }

};

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
//                items.insert(*std::prev(j->first.end()));
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

itemset_t L1(int minSupp, const transactions_t& transactions) {
    std::unordered_map<int, int> count;
    for (auto& transaction : transactions) {
        for (auto& item : transaction) {
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
//            L.insert(std::pair<items_t , int>(items, it->second));
        }
    }
    return L;
}

itemset_t generateL(itemset_t& C, int minSupp, const transactions_t& transactions) {
    for (auto& transaction : transactions){
        for (auto it = C.begin(); it != C.end(); ++it) {
            bool in = true;
            for (auto j = it->first.begin(); j != it->first.end(); ++j) {
                if (transaction.find(*j) == transaction.end()) {
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
    int minSupport = 10;
//    std::string filename("/Users/Shangtong/GitHub/DataMining/cmake-build-debug/dataS.txt");
    std::string filename("/Users/Shangtong/GitHub/DataMining/cmake-build-debug/in.txt");
    auto transactions = readData(filename);
    auto L = L1(minSupport, transactions);
    while (true) {
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
        L = generateL(C, minSupport, transactions);
    }
    return 0;
}