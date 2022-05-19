#ifndef CALCULATION_CUH
#define CALCULATION_CUH

#include <cuda_runtime.h>

#include "matrix.cuh"

template <class T>
struct comparator {
    __host__ __device__ bool operator()(T a, T b) {
        return abs(a) < abs(b);
    }
};

template <class T>
__global__ void SwapRows(T* dev_matrix, int n, int r1, int r2) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int offsetx = blockDim.x * gridDim.x;
    for (int i = idx; i < 2 * n; i += offsetx) {
        T tmp = dev_matrix[r1 + n * i];

        dev_matrix[r1 + n * i] = dev_matrix[r2 + n * i];
        dev_matrix[r2 + n * i] = tmp;
    }
}

template <class T>
__global__ void ForwardGauss(T* dev_matrix, int n, int i) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy + i + 1; k < 2 * n; k += offsety)
        for (int j = idx + i + 1; j < n; j += offsetx)
            dev_matrix[k * n + j]  -= (dev_matrix[k * n + i] * dev_matrix[i * n + j] / dev_matrix[i + i * n]);
}

template <class T>
__global__ void BackwardGauss(T* dev_matrix, int n, int i) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int k = idy + i + 1; k < 2 * n; k += offsety)
        for (int j = i - 1 - idx; j >= 0; j -= offsetx)
            dev_matrix[k * n + j]  -= (dev_matrix[k * n + i] * dev_matrix[i * n + j] / dev_matrix[i + i * n]);
}

template <class T>
__global__ void Normalize(T* dev_matrix, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int i = idy; i < n; i += offsety)
        for (int j = n + idx; j < 2 * n; j += offsetx)
            dev_matrix[i + j * n] /= dev_matrix[i + i * n];
}


TMatrix LU_Solving_System(TMatrix const& L, TMatrix const& U, TMatrix b, std::vector<std::pair<size_t, size_t>> const& p);
long double LU_Determinant(TMatrix const& U, std::vector<std::pair<size_t, size_t>> const&);
TMatrix LU_Inverse_Matrix(TMatrix const& L, TMatrix const& U, std::vector<std::pair<size_t, size_t>> const& p);

#endif