cmake_minimum_required(VERSION 3.16.3)

project(cuda_projct CUDA)

set(CMAKE_CUDA_STANDARD  14)
add_compile_options(-G)



add_executable(main main.cu
        src/matrix.cu
        src/skeleton.cu
        src/core.cu
        src/tensor_train.cu
        src/implicit_tensor.cu
        src/unfolding_matrix.cu
        src/tensor.cu
        src/calculation.cu
        src/host_matrix.cu)

target_link_libraries(main -lcublas)