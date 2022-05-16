#ifndef CALCULATION_CUH
#define CALCULATION_CUH

#include <cuda_runtime.h>

template <class T>
struct comparator {
    __host__ __device__ bool operator()(T& a, T& b) {
        return abs(a) < abs(b);
    }
};
template <class T>
__global__ void SwapRows(T* dev_matrix, size_t n, size_t r1, size_t r2);

template <class T>
__global__ void ForwardGauss(T* dev_matrix, size_t n, size_t i);

template <class T>
__global__ void BackwardGauss(T* dev_matrix, size_t n, size_t i);

template <class T>
__global__ void Normalize(T* dev_matrix, size_t n);
#endif
