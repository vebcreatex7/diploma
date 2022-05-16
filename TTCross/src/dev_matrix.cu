#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#include "../include/dev_matrix.cuh"
#include "../include/calculation.cuh"

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
DevMatrix<T>::DevMatrix(const Matrix<T>& other) : rows_(other.rows_), cols_(other.cols_) {
	size_t bytes = sizeof(T) * rows_ * cols_;

    CudaErrHandler(cudaMalloc((void**)&devData_, bytes));
    CudaErrHandler(cudaMemcpy(devData_, other.data_, bytes, cudaMemcpyHostToDevice));
}


template <class T>
DevMatrix<T>::~DevMatrix() {
	rows_ = cols_ = 0;
    CudaErrHandler(cudaFree(devData_));
}

template <class T>
Matrix<T> DevMatrix<T>::ToHost() const {
	Matrix<T> res(rows_, cols_);
	size_t bytes = sizeof(T) * rows_ * cols_;

	cudaMemcpy(res.data_, devData_, bytes, cudaMemcpyDeviceToHost);

	return res;
}

template <class T>
DevMatrix<T> DevMatrix<T>::Inverse() const {
	size_t n = cols_;
	size_t bytes = sizeof(T) * n * n;
	T* devInverse;

	cudaMalloc((void**)&devInverse, 2 * bytes);
	cudaMemset(devInverse, 0, 2 * bytes);
	cudaMemcpy(devInverse, bytes, cudaMemcpyDeviceToDevice);

	comparator<T> comp;

	for (size_t i = 0; i < n; i++) {
		thrust::device_ptr<T> thrustInverse = thrust::device_pointer_cast(devInverse);
		thrust::device_ptr<T> max = thrust::max_element(thrustInverse + i * (1 + n), thrustInverse + n * (i + 1), comp);
		size_t maxIdx = max - (thrustInverse + i * n);

		if (maxIdx != i)
			SwapRows<T><<<dim3(xBlocks), dim3(xThreads)>>>(devInverse, n, i, maxIdx);

		ForwardGauss<T><<<dim3(xBlocks, yBlocks), dim3(xThreads,yThreads)>>>(devInverse,n,i);
	}

	for (size_t i = n - 1; i != (size_t)-1; i--) {
		BackwardGauss<T><<<dim3(xBlocks, yBlocks), dim3(xThreads,yThreads)>>>(devInverse, n, i);
	}

	Normalize<T><<<dim3(xBlocks, yBlocks), dim3(xThreads,yThreads)>>>(devInverse, n);

	DevMatrix<T> res(n,n);

	cudaMemcpy(res.devData_, devInverse + bytes, bytes, cudaMemcpyDeviceToDevice);

	cudaFree(devInverse);

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

template <class T>
DevMatrix<T>& DevMatrix<T>::operator= (const DevMatrix<T>& other) {
	this->~DevMatrix();

	rows_ = other.rows_;
	cols_ = other.cols_;

	size_t bytes = sizeof(T) * rows_ * cols_;

	CudaErrHandler(cudaMalloc((void**)&devData_, bytes));
	CudaErrHandler(cudaMemcpy(devData_, other.devData_, bytes, cudaMemcpyDeviceToDevice));

	return *this;
}

template <class T>
DevMatrix<T>& DevMatrix<T>::operator= (const Matrix<T>& other) {
	this->~DevMatrix();

	rows_ = other.rows_;
	cols_ = other.cols_;

	size_t bytes = sizeof(T) * rows_ * cols_;

	CudaErrHandler(cudaMalloc((void**)&devData_, bytes));
	CudaErrHandler(cudaMemcpy(devData_, other.data_, bytes, cudaMemcpyHostToDevice));

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

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, cols_, &alpha, devData_, m, other.devData_, cols_, &beta, res.devData_, m);

	return res;
}