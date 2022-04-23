cmake_minimum_required(VERSION 3.16.3)

project(diploma)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall")
add_executable(main matrix.cpp main.cpp calculation.cpp skeleton.cpp)

