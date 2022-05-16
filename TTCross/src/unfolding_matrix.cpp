#include "../include/unfolding_matrix.hpp"


UnfoldingMatrix::UnfoldingMatrix() {
    rows_ = cols_ = 0;
    t_ = nullptr;
}

UnfoldingMatrix::UnfoldingMatrix(ImplicitTensor& t, size_t rows, size_t cols) :
    rows_(rows), cols_(cols), t_(&t) {}


double UnfoldingMatrix::operator()(size_t i, size_t j) const {
    return t_->operator()(i,j);
}

UnfoldingMatrix& UnfoldingMatrix::operator= (const UnfoldingMatrix& other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    t_ = other.t_;

    return *this;
}

TMatrix UnfoldingMatrix::ExplicitRows(const std::vector<size_t>& I) const {
    size_t rows = I.size();
    TMatrix res(rows, cols_, 0.);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols_; j++) {
            res[i][j] = t_->operator()(I[i],j);
        }
    }

    return res;
}

TMatrix UnfoldingMatrix::ExplicitCols(const std::vector<size_t>& J) const {
    size_t cols = J.size();
    TMatrix res(rows_, cols ,0.);

    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols; j++) {
            res[i][j] = t_->operator()(i, J[j]);
        }
    }

    return res;
}

TMatrix UnfoldingMatrix::ExplicitMaxvol(const std::vector<size_t>& I,const std::vector<size_t>& J) const {
    size_t n = I.size();
    TMatrix res(n,n,0.);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            res[i][j] = t_->operator()(I[i],J[j]);
        }
    }

    return res;
}

template <class T> Matrix<T> UnfoldingMatrix::GetRows(const std::vector<size_t>& I) const {
    size_t rows = I.size();
    Matrix<T> res(rows, cols_);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols_; j++) {
            res(i,j) = t_->operator()(I[i],j);
        }
    }

    return res;
}

template <class T> Matrix<T> UnfoldingMatrix::GetCols(const std::vector<size_t>& J) const {
    size_t cols = J.size();
    Matrix<T> res(rows_, cols);

    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols; j++) {
            res(i,j) = this->operator()(i, J[j]);
        }
    }

    return res;
}

template <class T> Matrix<T> UnfoldingMatrix::GetMaxvol(const std::vector<size_t> &I,
                                                        const std::vector<size_t> &J) const {
    size_t n = I.size();
    Matrix<T> res(n, n);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            res(i,j) = this->operator()(I[i],J[j]);
        }
    }

    return res;
}

void UnfoldingMatrix::print() const {
    std::cout << *this;
}

std::ostream& operator<< (std::ostream& os, const UnfoldingMatrix& A) {
    os << std::setprecision(5) << std::fixed;
    for (size_t i = 0; i < A.rows_; i++) {
        for (size_t j = 0; j < A.cols_; j++) {
            os << A(i,j) << ' ';
        }
        os << std::endl;
    }

    return os;
}

