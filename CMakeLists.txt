cmake_minimum_required(VERSION 3.6)
project(DataMining)

set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(SOURCE_FILES APRIORI/apriori.cpp)
add_executable(APRIORI ${SOURCE_FILES})