#include "core.hpp"

Core::Core() {
    a_ = b_ = c_ = 0;
    data_ = nullptr;
}

Core::Core(const TMatrix& A, size_t r_prev, size_t n_k, size_t r_k) {
    a_ = r_prev, b_ = n_k, c_ = r_k;
    data_ = new double[a_ * b_ * c_];

    for (size_t i = 0; i < a_; i++) {
        for (size_t j = 0; j < b_; j++) {
            for (size_t k = 0; k < c_; k++) {
                size_t p = i * b_ * c_ + j * c_ + k;

                data_[p] = A[p / A.Get_Cols()][p % A.Get_Cols()];
            }
        }
    }
}

Core::Core(const Core& other) {
    a_ = other.a_, b_ = other.b_, c_ = other.c_;
    data_ = new double[a_ * b_ * c_];

    for (size_t i = 0; i < a_*b_*c_; i++) {
        data_[i] = other.data_[i];
    }
}

Core::~Core() {delete []data_;}

std::tuple<size_t,size_t,size_t> Core::Sizes() const {
    return make_tuple(a_, b_, c_);
}

double Core::operator()(size_t i, size_t j, size_t k) const {
    return data_[i * b_ * c_ + j * c_ + k];
}

Core& Core::operator= (const Core& other) {
    this->~Core();

    a_ = other.a_;
    b_ = other.b_;
    c_ = other.c_;

    data_ = new double[a_ * b_ * c_];
    for (size_t i = 0; i < a_*b_*c_; i++) {
        data_[i] = other.data_[i];
    }

    return *this;
}

TMatrix Core::operator()(size_t j) const {
    TMatrix res(a_, c_, 0.);

    for (size_t i = 0; i < a_; i++) {
        for (size_t k = 0; k < c_; k++) {
            res[i][k] = this->operator()(i,j,k);
        }
    }

    return res;
}

std::ostream& operator<< (std::ostream& out, const Core& c) {
    out << std::setprecision(5) << std::fixed;

    for (size_t j = 0; j < c.b_; j++) {
        for (size_t i = 0; i < c.a_; i++) {
            for (size_t k = 0; k < c.c_; k++) {
                out << c(i,j,k) << ' ';
            }
            out << '\t';
        }
        out << std::endl;
    }

    return out;
}

void Core::print() const {
    std::cout << *this;
}