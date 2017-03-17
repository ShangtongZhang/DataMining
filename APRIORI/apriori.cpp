#include "iomanip"
#include "iostream"
#include "fstream"
#include "vector"
#include "string"
#include "sstream"
#include "unordered_map"
#include "unordered_set"
#include "set"
#include "map"
#include "mutex"
#include "atomic"
#include "thread"
#include "algorithm"

using items_t = std::vector<int>;
// how should hash table be slower than rb tree
//using itemset_t = std::unordered_map<items_t, int>;
//using transaction_t = std::unordered_set<int>;
using itemset_t = std::map<items_t, int>;
using transaction_t = std::set<int>;
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

class AsyncTransactionFeeder {
public:
    const int poolSize;
    std::string filename;
    std::fstream file;
    transactions_t transactionsPool;
    std::atomic<int> poolInPos;
    std::atomic<int> poolOutPos;
    std::atomic<bool> dataEnd;
    std::atomic<bool> fitInMem;

    AsyncTransactionFeeder() :
            poolSize(10000000),
            transactionsPool(poolSize) {
        poolInPos = 0;
        poolOutPos = 0;
        dataEnd = false;
        fitInMem = true;
    }

    bool dataInMem() {
//        return false;
        return dataEnd && fitInMem;
    }

    size_t dataSize() {
        return poolInPos;
    }

    void attachFile(const std::string& filename) {
        this->filename = filename;
        file.open(filename);
    }

    void reset() {
        if (fitInMem && dataEnd) {
            poolOutPos = 0;
            dataEnd = true;
            return;
        }
        file.close();
        file.open(filename);
        poolInPos = poolOutPos = 0;
        dataEnd = false;
        transactionsPool.clear();
        transactionsPool.resize(poolSize);
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
            while ((poolInPos + 1) % poolSize == poolOutPos) {}
            auto& transaction = transactionsPool[poolInPos];
            transaction.clear();
            int item;
            while (ss >> item) {
                transaction.insert(item);
            }
            if (poolInPos + 1 == poolSize) {
                fitInMem = false;
            }
            poolInPos = (poolInPos + 1) % poolSize;
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
            poolOutPos = (poolOutPos + 1) % poolSize;
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

itemset_t L1(float minSuppRatio, float& minSupp, int& nTransactions) {
    std::unordered_map<int, int> count;
    feeder.reset();
    feeder.start();
    while (true) {
        auto pTransaction = feeder.getNextTransaction();
        if (!pTransaction) {
            break;
        }
        nTransactions++;
        for (auto& item : *pTransaction) {
            if (count.find(item) != count.end()) {
                count[item] += 1;
            } else {
                count[item] = 1;
            }
        }
    }
    minSupp = nTransactions * minSuppRatio;
    itemset_t L;
    for (auto it = count.begin(); it != count.end(); ++it) {
        if (it->second >= minSupp) {
            std::vector<int> items{it->first};
            L[items] = it->second;
        }
    }
    return L;
}

void countItems(size_t startPoint, size_t endPoint, itemset_t* C, std::mutex* m) {
    for (size_t i = startPoint; i < endPoint; ++i) {
        auto& transaction = feeder.transactionsPool[i];
        for (auto it = C->begin(); it != C->end(); ++it) {
            bool in = true;
            for (auto j = it->first.begin(); j != it->first.end(); ++j) {
                if (transaction.find(*j) == transaction.end()) {
                    in = false;
                    break;
                }
            }
            if (in) {
                m->lock();
                it->second += 1;
                m->unlock();
            }
        }
    }
}

itemset_t generateL(itemset_t& C, float minSupp) {
    feeder.reset();
    feeder.start();
    if (feeder.dataInMem()) {
        int factor = std::thread::hardware_concurrency();
        size_t len = feeder.dataSize();
        std::mutex mUpdate;
        std::vector<size_t > startPoint(factor);
        std::vector<size_t > endPoint(factor);
        size_t step = len / factor;
        for (int i = 0; i < factor; ++i) {
            startPoint[i] = step * i;
            endPoint[i] = step * (i + 1);
        }
        endPoint.back() = len;
        std::vector<std::thread> counters(factor);
        for (int i = 0; i < factor; ++i) {
            counters.emplace(counters.begin() + i, std::thread(countItems, startPoint[i], endPoint[i], &C, &mUpdate));
        }
        for (auto& t : counters) {
            if (t.joinable()) {
                t.join();
            }
        }
    } else {
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
    }
    itemset_t L;
    for (auto it = C.begin(); it != C.end(); ++it) {
        if (it->second >= minSupp) {
            L.insert(*it);
        }
    }
    return L;
}

std::vector<itemset_t> frequentItemsets;
std::vector<items_t> ruleLHS;
std::vector<items_t> ruleRHS;
std::vector<int> ruleSupp;
std::vector<float> ruleConf;

std::ostream& operator << (std::ostream& out, const items_t& items) {
    for (auto it = items.begin(); it != items.end(); ++it) {
        if (it != items.begin()) {
            out << ",";
        }
        out << *it;
    }
    return out;
}

void deriveRuleImpl(std::vector<bool>& bit, int cur, const items_t& items, float minConf) {
    if (cur == bit.size()) {
        items_t lhs, rhs;
        for (int i = 0; i < (int)bit.size(); ++i) {
            if (bit[i]) {
                lhs.push_back(items[i]);
            } else {
                rhs.push_back(items[i]);
            }
        }
        if (lhs.empty() || lhs.size() == items.size()) {
            return;
        }
        float conf = (float)frequentItemsets[items.size()][items] / frequentItemsets[lhs.size()][lhs];
        if (conf >= minConf) {
//            std::cout << lhs << " -> " << rhs << std::endl;
            ruleLHS.push_back(std::move(lhs));
            ruleRHS.push_back(std::move(rhs));
            ruleSupp.push_back(frequentItemsets[items.size()][items]);
            ruleConf.push_back(conf);
        }
    } else {
        deriveRuleImpl(bit, cur + 1, items, minConf);
        bit[cur] = true;
        deriveRuleImpl(bit, cur + 1, items, minConf);
        bit[cur] = false;
    }
}

void deriveRule(const items_t& items, float minConf) {
    std::vector<bool> bit(items.size(), false);
    deriveRuleImpl(bit, 0, items, minConf);
}

void generateStrongRules(float minConf) {
    for (int k = 2; k < (int)frequentItemsets.size(); ++k) {
        for (auto it = frequentItemsets[k].begin(); it != frequentItemsets[k].end(); ++it) {
            deriveRule(it->first, minConf);
        }
    }
}

void showFrequentItemsets(int nTransactions) {
    for (int k = 1; k < (int)frequentItemsets.size(); ++k) {
        for (auto& it : frequentItemsets[k]) {
            std::cout << it.first << " (" << std::fixed << std::setprecision(2)
                      << it.second / (float)nTransactions << ")" << std::endl;
        }
    }
}

void showStrongRules(int nTransactions) {
    for (int i = 0; i < (int)ruleConf.size(); ++i) {
        std::cout << ruleLHS[i] << " -> " << ruleRHS[i]
                  << " (" << std::fixed << std::setprecision(2)
                  << ruleSupp[i] / (float)nTransactions << "," << std::fixed << std::setprecision(2)
                  << ruleConf[i] << ")" << std::endl;
    }
}

int main(int argc, char *argv[]) {
    std::string filename(argv[1]);
    float minSuppRatio = std::stof(argv[2]);
    float minConf = std::stof(argv[3]);
    char op = 'e';
    if (argc == 5) {
        op = argv[4][0];
    }
    float minSupp;
    int nTransactions = 0;
//    std::string filename("/Users/Shangtong/GitHub/DataMining/cmake-build-debug/data.txt");
//    std::string filename("/Users/Shangtong/GitHub/DataMining/cmake-build-debug/in.txt");
    feeder.attachFile(filename);
    frequentItemsets.resize(1);
    auto L = L1(minSuppRatio, minSupp, nTransactions);
    while (true) {
//        std::cout << "Freq" << std::endl;
//        for (auto it = L.begin(); it != L.end(); ++it) {
//            for (auto& item : it->first) {
//                std::cout << item << " ";
//            }
//            std::cout << "-> " << it->second << std::endl;
//        }
//        std::cout << L.size() << std::endl;
        if (L.empty()) {
            break;
        }
        auto C = generateC(L);
        frequentItemsets.push_back(std::move(L));
//        std::cout << "Candidate:" << std::endl;
//        for (auto it = C.begin(); it != C.end(); ++it) {
//            for (auto& item : it->first) {
//                std::cout << item << " ";
//            }
//            std::cout << "-> " << it->second << std::endl;
//        }
//        std::cout << C.size() << std::endl;
        L = generateL(C, minSupp);
    }
//    for (int i = 1; i < (int)frequentItemsets.size(); ++i) {
//        std::cout << "Number of frequent " << i << "_itemsets:" << frequentItemsets[i].size() << std::endl;
//    }
//    generateStrongRules(minConf);
//    std::cout << "Number of association rules " << ruleConf.size() << std::endl;
    if (op == 'r') {
        showStrongRules(nTransactions);
    } else if (op == 'f') {
        showFrequentItemsets(nTransactions);
    } else if (op == 'a') {
        showFrequentItemsets(nTransactions);
        showStrongRules(nTransactions);
    } else {
        std::cout << frequentItemsets.size() - 1 << " " << ruleConf.size() << std::endl;
    }
    return 0;
}