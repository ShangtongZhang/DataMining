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
/NEAT: Exploit NEAT to play OpenAI games and Super Mario

/Neat/neat: [neat-python](https://github.com/CodeReclaimers/neat-python)

/Neat/SuperMarioBros: [nes](https://github.com/zoq/nes)
>SuperMarioBros is copied from Marcus Edel, originally it's a C++ interface for the emulator fceux, I write python interface in nes.py

To make fceux work, please follow the [instructions](https://github.com/zoq/nes) from Marcus Edel.

**Notice that the default built-in version of lua of fceux is 5.1, you can either change the built-in version to your local version or make sure you install lua, luarocks and luasocket for 5.1 version.**

**Setting include path for new lua users are also tricky, you can set them as following:**

```bash
export LUA_PATH=";/usr/local/share/lua/5.1/?.lua;/path/to/SuperMarioBros/?.lua"
export LUA_CPATH=";/usr/local/lib/lua/5.1/?.so;"
```


