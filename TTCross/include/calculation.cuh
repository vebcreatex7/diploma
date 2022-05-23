#ifndef CALCULATION_CUH
#define CALCULATION_CUH

#include <cuda_runtime.h>

#include "matrix.cuh"

#define IDX2C(i,j,rows) (((j)*(rows)) + (i))

#define CudaErrHandler(call)  										\
do {																\
	cudaError_t zzz = call;											\
	if (zzz != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s Code: %d\n",	\
				__FILE__, __LINE__, cudaGetErrorString(zzz), zzz);	\
		exit(0);													\
	}																\
} while(0)

#define XTHREADS 128
#define YTHREADS 8

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
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t idy = threadIdx.y + blockDim.y * blockIdx.y;
    size_t offsetx = blockDim.x * gridDim.x;
    size_t offsety = blockDim.y * gridDim.y;

    for (size_t k = idy + i + 1; k < 2 * n; k += offsety)
        for (size_t j = idx + i + 1; j < n; j += offsetx)
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
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t idy = threadIdx.y + blockDim.y * blockIdx.y;
    size_t offsetx = blockDim.x * gridDim.x;
    size_t offsety = blockDim.y * gridDim.y;

    for (size_t i = idy; i < n; i += offsety)
        for (size_t j = n + idx; j < 2 * n; j += offsetx)
            dev_matrix[i + j * n] /= dev_matrix[i + i * n];
}

template <class T>
__global__ void Eye(T* dev_matrix, int n) {
    size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    size_t offsetx = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += offsetx) {
        dev_matrix[i + i * n] = (T)1;
    }
}

template <class T>
__global__ void matrixMul(const T *a, const T *b, T *c, size_t m, size_t n, size_t k)
{

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j =  blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= m || j >= k) return;

    T sum = 0;

    for (size_t l = 0; l < n; l++) {
        sum += a[IDX2C(i,l,m)] * b[IDX2C(l,j,n)];
    }

    c[IDX2C(i,j,m)] = sum;
}

template <class T>
void MatrixMul(const T* a, const T* b, T* c, size_t m, size_t n, size_t k, cudaStream_t s) {
    dim3 blocks = dim3(
            (int) std::ceil((double)m / XTHREADS),
            (int) std::ceil((double)k / YTHREADS),
            1
    );

    dim3 threads = dim3(
            XTHREADS,
            YTHREADS,
            1
    );
    matrixMul<<<blocks, threads,0,s>>>(a,b,c,m,n,k);
    CudaErrHandler(cudaGetLastError());
}

TMatrix LU_Solving_System(TMatrix const& L, TMatrix const& U, TMatrix b, std::vector<std::pair<size_t, size_t>> const& p);
long double LU_Determinant(TMatrix const& U, std::vector<std::pair<size_t, size_t>> const&);
TMatrix LU_Inverse_Matrix(TMatrix const& L, TMatrix const& U, std::vector<std::pair<size_t, size_t>> const& p);

#endif