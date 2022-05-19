#ifndef DEV_MATRIX_CUH
#define DEV_MATRIX_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#include "host_matrix.cuh"
#include "matrix.cuh"
#include "calculation.cuh"

#define CudaErrHandler(call)  										\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s Code: %d\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res), res);		\
		exit(0);													\
	}																\
} while(0)

const size_t xThreads = 128;
const size_t yThreads = 8;
const size_t xBlocks  = 128;
const size_t yBlocks  = 8;

template <class T>
class DevMatrix {
private:
    size_t rows_, cols_;
    T* devData_;



public:
    DevMatrix();
    DevMatrix(size_t rows, size_t cols, T val = 0);
    DevMatrix(const DevMatrix<T>& other);
    explicit DevMatrix(const Matrix& other);
    ~DevMatrix();
    Matrix ToHost() const;
    TMatrix ToTMatrix() const;
    DevMatrix<T> Inverse() const;
    size_t Rows() const;
    size_t Cols() const;
    const T* Data() const {return devData_;};
    void SetData(const T* src, size_t srcOffset, size_t dstOffset, size_t bytes);

    DevMatrix<T>& operator= (const DevMatrix<T>& other);
    DevMatrix<T>& operator= (const Matrix& other);
    DevMatrix<T> operator* (const DevMatrix<T>& other) const;

};

template <class T>
DevMatrix<T>::DevMatrix() : rows_(0), cols_(0), devData_(NULL) {}

template <class T>
DevMatrix<T>::DevMatrix(size_t rows, size_t cols, T val) : rows_(rows), cols_(cols) {
    size_t bytes = sizeof(T) * rows_ * cols_;

    CudaErrHandler(cudaMalloc((void**)&devData_, bytes));
    CudaErrHandler(cudaMemset(devData_, val, bytes));
}

template <class T>
DevMatrix<T>::DevMatrix(const DevMatrix<T>& other) : rows_(other.rows_), cols_(other.cols_) {
    size_t bytes = sizeof(T) * rows_ * cols_;

    CudaErrHandler(cudaMalloc((void**)&devData_, bytes));
    CudaErrHandler(cudaMemcpy(devData_, other.devData_, bytes, cudaMemcpyDeviceToDevice));
}

template <class T>
DevMatrix<T>::DevMatrix(const Matrix& other) : rows_(other.Rows()), cols_(other.Cols()) {
    size_t bytes = sizeof(T) * rows_ * cols_;

    CudaErrHandler(cudaMalloc((void**)&devData_, bytes));
    CudaErrHandler(cudaMemcpy(devData_, other.Data(), bytes, cudaMemcpyHostToDevice));
}


template <class T>
DevMatrix<T>::~DevMatrix() {
    rows_ = cols_ = 0;
    CudaErrHandler(cudaFree(devData_));
}

template <class T>
Matrix DevMatrix<T>::ToHost() const {
    Matrix res(rows_, cols_);
    size_t bytes = sizeof(T) * rows_ * cols_;

    cudaMemcpy(res.Data(), devData_, bytes, cudaMemcpyDeviceToHost);

    return res;
}

template <class T>
TMatrix DevMatrix<T>::ToTMatrix() const {
    TMatrix res(rows_, cols_);

    size_t bytes = sizeof(T) * rows_ * cols_;
    T* tmp = (T*)malloc(bytes);

    CudaErrHandler(cudaMemcpy(tmp, devData_, bytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            res[i][j] = tmp[IDX2C(i,j,rows_)];
        }
    }

    free(tmp);
    return res;
}

template <class T>
DevMatrix<T> DevMatrix<T>::Inverse() const {
    size_t n = cols_;
    size_t bytes = sizeof(T) * n * n;
    T* devInverse;

    DevMatrix<T> tmp(n, 2 * n);

    CudaErrHandler(cudaMalloc((void**)&devInverse, 2 * bytes));
    //CudaErrHandler(cudaMemset(devInverse, 0, 2 * bytes));
    CudaErrHandler(cudaMemcpy(devInverse, devData_, bytes, cudaMemcpyDeviceToDevice));
    comparator<T> comp;

    for (int i = 0; i < n; i++) {
        thrust::device_ptr<T> thrustInverse = thrust::device_pointer_cast(devInverse);
        thrust::device_ptr<T>  max = thrust::max_element(thrustInverse + i + i * n, thrustInverse + n + i * n, comp);

        int maxIdx = max - (thrustInverse + i * n);

        if (maxIdx != i) {
            SwapRows<T><<<dim3(xBlocks), dim3(xThreads)>>>(devInverse, n, i, maxIdx);
            CudaErrHandler(cudaDeviceSynchronize());
        }

        ForwardGauss<T><<<dim3(xBlocks, yBlocks), dim3(xThreads,yThreads)>>>(devInverse,n,i);
        CudaErrHandler(cudaDeviceSynchronize());
    }

    for (int i = n - 1; i >= 0; i--) {
        BackwardGauss<T><<<dim3(xBlocks, yBlocks), dim3(xThreads,yThreads)>>>(devInverse, n, i);
        CudaErrHandler(cudaDeviceSynchronize());
    }

    Normalize<T><<<dim3(xBlocks, yBlocks), dim3(xThreads,yThreads)>>>(devInverse, n);
    CudaErrHandler(cudaDeviceSynchronize());

    //CudaErrHandler(cudaDeviceSynchronize());

    DevMatrix<T> res(n,n);

    res.SetData(devInverse, n * n, 0, bytes);
    //cudaMemcpy(res.devData_, devInverse + n * n, bytes, cudaMemcpyDeviceToDevice);

    CudaErrHandler(cudaFree(devInverse));

    return res;

}

template <class T>
size_t DevMatrix<T>::Rows() const {
    return rows_;
}

template <class T>
size_t DevMatrix<T>::Cols() const {
    return cols_;
}

template<class T>
void DevMatrix<T>::SetData(const T* src, size_t srcOffset, size_t dstOffset, size_t bytes) {
    CudaErrHandler(cudaMemcpy(devData_ + dstOffset, src + srcOffset, bytes, cudaMemcpyDeviceToDevice));
}

template <class T>
DevMatrix<T>& DevMatrix<T>::operator= (const DevMatrix<T>& other) {
    if (this == &other) return *this;

    this->~DevMatrix();

    rows_ = other.rows_;
    cols_ = other.cols_;

    size_t bytes = sizeof(T) * rows_ * cols_;

    CudaErrHandler(cudaMalloc((void**)&devData_, bytes));
    CudaErrHandler(cudaMemcpy(devData_, other.devData_, bytes, cudaMemcpyDeviceToDevice));

    return *this;
}

template <class T>
DevMatrix<T>& DevMatrix<T>::operator= (const Matrix& other) {
    this->~DevMatrix();

    rows_ = other.Rows();
    cols_ = other.Cols();

    size_t bytes = sizeof(T) * rows_ * cols_;

    CudaErrHandler(cudaMalloc((void**)&devData_, bytes));
    CudaErrHandler(cudaMemcpy(devData_, other.Data(), bytes, cudaMemcpyHostToDevice));

    return *this;
}

template<class T>
DevMatrix<T> DevMatrix<T>::operator* (const DevMatrix<T>& other) const {
    size_t m = rows_;
    size_t n = other.cols_;
    DevMatrix<T> res(m,n);

    size_t bytes = sizeof(T) * m * n;

    cudaMalloc((void**)&res.devData_, bytes);

    cublasHandle_t handle;
    cublasCreate(&handle);

    T alpha = 1;
    T beta = 0;

    cublasSgemm(handle,
                CUBLAS_OP_N,CUBLAS_OP_N,
                m, n, cols_,
                &alpha,devData_,m,
                other.devData_, cols_,
                &beta, res.devData_, m);

    return res;
}


#endif