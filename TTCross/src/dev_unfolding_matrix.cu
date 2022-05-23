#include "../include/dev_unfolding_matrix.cuh"

DevUnfoldingMatrix::DevUnfoldingMatrix() {
    rows_ = cols_ = 0;
    t_ = nullptr;
}

DevUnfoldingMatrix::DevUnfoldingMatrix(DevImplicitTensor &t, size_t rows, size_t cols) :
    rows_(rows), cols_(cols), t_(&t) {}

double *DevUnfoldingMatrix::ExplicitRows(const std::vector<size_t> &I) const {
    size_t m = I.size();
    size_t n = cols_;

    size_t bytes = sizeof(double) * m * n;

    double* res;
    double* devRes;

    res = (double*)malloc(bytes);
    CudaErrHandler(cudaMalloc((void**)&devRes, bytes));

    for (size_t i = 0; i < m; i ++) {
        for (size_t j = 0; j < n; j++) {
            res[IDX2C(i,j,m)] = (*t_)(I[i],j);
        }
    }

    CudaErrHandler(cudaMemcpy(devRes, res, bytes, cudaMemcpyHostToDevice));

    free(res);

    return devRes;
}

double *DevUnfoldingMatrix::ExplicitCols(const std::vector<size_t> &J) const {
    size_t m = rows_;
    size_t n = J.size();

    size_t bytes = sizeof(double) * m * n;

    double* res;
    double* devRes;

    res = (double*)malloc(bytes);
    CudaErrHandler(cudaMalloc((void**)&devRes, bytes));

    for (size_t i = 0; i < m; i ++) {
        for (size_t j = 0; j < n; j++) {
            res[IDX2C(i,j,m)] = (*t_)(i, J[j]);
        }
    }

    CudaErrHandler(cudaMemcpy(devRes, res, bytes, cudaMemcpyHostToDevice));

    free(res);

    return devRes;
}

double *DevUnfoldingMatrix::ExplicitMaxvol(const std::vector<size_t> &I, const std::vector<size_t> &J) const {
    size_t m = I.size();
    size_t n = J.size();

    size_t bytes = sizeof(double) * m * n;

    double* res;
    double* devRes;

    res = (double*)malloc(bytes);
    CudaErrHandler(cudaMalloc((void**)&devRes, bytes));

    for (size_t i = 0; i < m; i ++) {
        for (size_t j = 0; j < n; j++) {
            res[IDX2C(i,j,m)] = (*t_)(I[i], J[j]);
        }
    }

    CudaErrHandler(cudaMemcpy(devRes, res, bytes, cudaMemcpyHostToDevice));

    free(res);

    return devRes;
}

void DevUnfoldingMatrix::print() const {
    std::cout << *this;
}

double DevUnfoldingMatrix::operator()(size_t i, size_t j) const {
    return (*t_)(i,j);
}

DevUnfoldingMatrix &DevUnfoldingMatrix::operator=(const DevUnfoldingMatrix &other) {
    if (this == &other) return *this;

    rows_ = other.rows_;
    cols_ = other.cols_;
    t_ = other.t_;

    return *this;
}

std::ostream &operator<<(std::ostream &os, const DevUnfoldingMatrix &m) {
    os << std::setprecision(5) << std::fixed;

    for (size_t i = 0; i < m.rows_; i++) {
        for (size_t j = 0; j < m.cols_; j++) {
            os << m(i,j) << ' ';
        }

        os << std::endl;
    }

    return os;
}




