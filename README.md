>My work for CMPUT 690 Data Mining, Winter 2017, UAlberta
# Apriori
/APRIORI/apriori.cpp: A C++ implementation of Apriori algorithm
* asynchronous data feeder
* multi-thread for generating frequent itemsets
>The executable should get four parameters as input: 1- file name; 2-minimum-support; and 3- minimum confidence. The thresholds should be numbers between 0 and 1. The forth parameter is either "r", "f", "a", or absent. When "r", then all strong association rules are displayed. When "f" then all frequent itemsets are displayed. When "a" then all frequent itemsets and all strong association rules are displayed. When absent, then only the number of frequent itemsets of different sizes and the number of strong rules are displayed
```bash
./apriori data.txt 0.001 0.8 a
```
# NEAT
/NEAT-OpenAI: Exploit NEAT to play OpenAI games
>Dependency: [neat-python](https://github.com/CodeReclaimers/neat-python) [OpenAI-gym](https://gym.openai.com/)

