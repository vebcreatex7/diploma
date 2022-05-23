#include "../include/dev_core.cuh"
#include "../include/calculation.cuh"

DevCore::DevCore() {
    a_ = b_ = c_ = 0;
}

DevCore::DevCore(const double *src, size_t r_prev, size_t n_k, size_t r_k) {
    a_ = r_prev, b_ = n_k, c_ = r_k;

    size_t bytes = sizeof(double) * a_ * b_ * c_;
    CudaErrHandler(cudaMalloc((void**)&devData_, bytes));
    CudaErrHandler(cudaMemset(devData_,0,bytes));


    for (size_t i = 0; i < r_k; i++) {
        for (size_t j = 0; j < n_k; j++) {
            size_t srcOffset = (i * r_prev * n_k + j * r_prev);
            size_t dstOffset = (i * r_prev + j * r_prev * r_k);
            size_t size = sizeof(double) * r_prev;

            CudaErrHandler(cudaMemcpy(devData_ + dstOffset, src + srcOffset, size, cudaMemcpyDeviceToDevice));
        }
    }
}

DevCore::DevCore(const DevCore &other) {
    a_ = other.a_, b_ = other.b_, c_ = other.c_;

    size_t bytes = sizeof(double) * a_ * b_ * c_;
    CudaErrHandler(cudaMalloc((void**)&devData_, bytes));

    CudaErrHandler(cudaMemcpy(devData_, other.devData_, bytes, cudaMemcpyDeviceToDevice));
}

DevCore::~DevCore() {
    CudaErrHandler(cudaFree(devData_));
}

double *DevCore::Matrix(size_t i) const {
    return devData_ + i * a_ * c_;
}

DevCore& DevCore::operator=(const DevCore &other) {
    if (this == &other) return *this;

    this->~DevCore();

    a_ = other.a_;
    b_ = other.b_;
    c_ = other.c_;

    size_t bytes = sizeof(double) * a_ * b_ * c_;

    CudaErrHandler(cudaMemcpy(devData_, other.devData_, bytes, cudaMemcpyDeviceToDevice));

    return *this;
}




