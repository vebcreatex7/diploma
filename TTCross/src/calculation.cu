#include "../include/calculation.cuh"

template <class T>
__global__ void SwapRows(T* dev_matrix, size_t n, size_t r1, size_t r2) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t offsetx = blockDim.x * gridDim.x;
    for (size_t i = idx; i < 2 * n; i += offsetx) {
        T tmp = dev_matrix[r1 + n * i];
        dev_matrix[r1 + n * i] = dev_matrix[r2 + n * i];
        dev_matrix[r2 + n * i] = tmp;
    }
}

template <class T>
__global__ void ForwardGauss(T* dev_matrix, size_t n, size_t i) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t idy = threadIdx.y + blockDim.y * blockIdx.y;
    size_t offsetx = blockDim.x * gridDim.x;
    size_t offsety = blockDim.y * gridDim.y;

    for (size_t k = idy + i + 1; k < 2 * n; k += offsety)
        for (size_t j = idx + i + 1; j < n; j += offsetx)
            dev_matrix[k * n + j]  -= (dev_matrix[k * n + i] * dev_matrix[i * n + j] / dev_matrix[i + i * n]);
}

template <class T>
__global__ void BackwardGauss(T* dev_matrix, size_t n, size_t i) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t idy = threadIdx.y + blockDim.y * blockIdx.y;
    size_t offsetx = blockDim.x * gridDim.x;
    size_t offsety = blockDim.y * gridDim.y;
    
    for (size_t k = idy + i + 1; k < 2 * n; k += offsety)
        for (size_t j = i - 1 - idx; j >= (size_t)-1; j -= offsetx)
            dev_matrix[k * n + j]  -= (dev_matrix[k * n + i] * dev_matrix[i * n + j] / dev_matrix[i + i * n]);
}

template <class T>
__global__ void Normalize(T* dev_matrix, size_t n) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t idy = threadIdx.y + blockDim.y * blockIdx.y;
    size_t offsetx = blockDim.x * gridDim.x;
    size_t offsety = blockDim.y * gridDim.y;

    for (size_t i = idy; i < n; i += offsety)
        for (size_t j = n + idx; j < 2 * n; j += offsetx)
            dev_matrix[i + j * n] /= dev_matrix[i + i * n];
}