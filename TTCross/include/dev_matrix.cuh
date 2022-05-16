#ifndef DEV_MATRIX_CUH
#define DEV_MATRIX_CUH

#include <stdlib.h>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "host_matrix.hpp"

#define CudaErrHandler(call)  										\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

#define CublasErrHandler(call)                                      \
do {                                                                \
    cublasStatus_t res = call;                                      \
    if (res != CUBLAS_STATUS_SUCCESS) {                             \
        fprintf(stderr, "ERROR in %s:%d. Code: %d\n",               \
                __FILE__, __LINE__, res));                          \
            exit(0);                                                \
    }                                                               \
} while(0)

#define IDX2C(i,j,rows) (((j)*(rows)) + (i))

const size_t xThreads = 32;
const size_t yThreads = 128;
const size_t xBlocks  = 32;
const size_t yBlocks  = 128;

template <class T>
class DevMatrix {
private:
    size_t rows_, cols_;
    T* devData_;

    friend Matrix<T>;

public:
    DevMatrix();
    DevMatrix(size_t rows, size_t cols, T val = 0);
    DevMatrix(const DevMatrix<T>& other);
    DevMatrix(const Matrix<T>& other);
    ~DevMatrix();
    Matrix<T> ToHost() const;
    DevMatrix<T> Inverse() const;
    size_t Rows() const;
    size_t Cols() const;

    __host__ 
    DevMatrix<T>& operator= (const DevMatrix<T>& other);
    DevMatrix<T>& operator= (const Matrix<T>& other);
    DevMatrix<T> operator* (const DevMatrix<T>& other) const;

};

#endif