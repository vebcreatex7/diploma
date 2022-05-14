cmake_minimum_required(VERSION 3.16.3)

project(diploma)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -ggdb")
set(CMAKE_BUILD_TYPE Debug)
add_executable(main main.cpp
                    src/matrix.cpp
                    src/calculation.cpp
                    src/skeleton.cpp
                    src/core.cpp
                    src/tensor_train.cpp
                    src/implicit_tensor.cpp
                    src/unfolding_matrix.cpp
                    src/tensor.cpp)

