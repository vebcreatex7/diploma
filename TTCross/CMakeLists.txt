cmake_minimum_required(VERSION 3.16.3)

project(diploma)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -ggdb")
set(CMAKE_BUILD_TYPE Debug)
add_executable(main main.cpp matrix.cpp calculation.cpp skeleton.cpp core.cpp tensorTrain.cpp implicitTensor.cpp unfoldingMatrix.cpp)

