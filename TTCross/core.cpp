#include "core.hpp"

Core::Core(TMatrix A, size_t r_prev, size_t n_k, size_t r_k) {
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

double Core::operator()(size_t i, size_t j, size_t k) const {
    return data_[i * b_ * c_ + j * c_ + k];
}