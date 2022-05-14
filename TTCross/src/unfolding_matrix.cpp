#include "../include/unfolding_matrix.hpp"


UnfoldingMatrix::UnfoldingMatrix() {
    n_ = m_ = 0;
    t_ = nullptr;
}

UnfoldingMatrix::UnfoldingMatrix(ImplicitTensor& t, size_t n, size_t m) :
    n_(n), m_(m), t_(&t) {}


double UnfoldingMatrix::operator()(size_t i, size_t j) const {
    return t_->operator()(i,j);
}

UnfoldingMatrix& UnfoldingMatrix::operator= (const UnfoldingMatrix& other) {
    n_ = other.n_;
    m_ = other.m_;
    t_ = other.t_;

    return *this;
}

TMatrix UnfoldingMatrix::ExplicitRows(const std::vector<size_t>& I) const {
    size_t n = I.size();
    TMatrix res(n,m_,0.);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m_; j++) {
            res[i][j] = t_->operator()(I[i],j);
        }
    }

    return res;
}

TMatrix UnfoldingMatrix::ExplicitCols(const std::vector<size_t>& J) const {
    size_t m = J.size();
    TMatrix res(n_,m,0.);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m; j++) {
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

void UnfoldingMatrix::print() const {
    std::cout << *this;
}

std::ostream& operator<< (std::ostream& os, const UnfoldingMatrix& A) {
    os << std::setprecision(5) << std::fixed;
    for (size_t i = 0; i < A.n_; i++) {
        for (size_t j = 0; j < A.m_; j++) {
            os << A(i,j) << ' ';
        }
        os << std::endl;
    }

    return os;
}

