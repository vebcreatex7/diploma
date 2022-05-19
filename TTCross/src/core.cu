#include "../include/core.cuh"

Core::Core() {
    a_ = b_ = c_ = 0;
    data_ = nullptr;
}

Core::Core(const DevMatrix<float> &A, size_t r_prev, size_t n_k, size_t r_k) {
    a_ = r_prev, b_ = n_k, c_ = r_k;
    matrices = std::vector<DevMatrix<float>>(n_k, DevMatrix<float>(r_prev, r_k));

    for (size_t i = 0; i < r_k; i++) {
        for (size_t j = 0; j < n_k; j++) {
            size_t srcOffset = sizeof(float) * (i * r_prev * n_k + j * r_prev);
            size_t dstOffset = sizeof(float) * i * r_prev;
            size_t size = sizeof(float) * r_prev;
            matrices[i].SetData(A.Data(), srcOffset, dstOffset, size);
        }
    }
}

Core::Core(const TMatrix& A, size_t r_prev, size_t n_k, size_t r_k) {
    a_ = r_prev, b_ = n_k, c_ = r_k;
    data_ = new float[a_ * b_ * c_];

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
    data_ = new float[a_ * b_ * c_];

    for (size_t i = 0; i < a_*b_*c_; i++) {
        data_[i] = other.data_[i];
    }
}

Core::~Core() {delete []data_;}

std::tuple<size_t,size_t,size_t> Core::Sizes() const {
    return std::make_tuple(a_, b_, c_);
}

void Core::SetMatrices(const DevMatrix<float>& A,  size_t r_prev, size_t n_k, size_t r_k) {
    matrices = std::vector<DevMatrix<float>>(n_k, DevMatrix<float>(r_prev, r_k));
    size_t size = r_prev;
    size_t srcOffset = 0;
    size_t dstOffset = 0;

    for (size_t i = 0; i < r_k; i++) {
        for (size_t j = 0; j < n_k; j++) {
            size_t bytes = sizeof(float) * r_prev;

            matrices[j].SetData(A.Data(), srcOffset, dstOffset, bytes);

            srcOffset += size;
        }

        dstOffset += r_prev;
    }
}

DevMatrix<float> Core::Matrix(size_t i) const {
    return matrices[i];
}

double Core::operator()(size_t i, size_t j, size_t k) const {
    return data_[i * b_ * c_ + j * c_ + k];
}

Core& Core::operator= (const Core& other) {
    if (this == &other) return *this;

    this->~Core();

    a_ = other.a_;
    b_ = other.b_;
    c_ = other.c_;

    data_ = new float[a_ * b_ * c_];
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
