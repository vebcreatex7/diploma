#include "../include/host_matrix.cuh"



Matrix::Matrix() : rows_(0), cols_(0), data_(nullptr) {}


Matrix::Matrix(size_t rows, size_t cols, float val) : rows_(rows), cols_(cols) {
    size_t bytes = sizeof(float) * rows_ * cols_;

    data_ = (float*)malloc(bytes);
}


Matrix::Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_) {
    size_t bytes = sizeof(float) * rows_ * cols_;
    data_ = (float*)malloc(bytes);
    memcpy(data_, other.data_, bytes);
}


Matrix::~Matrix() {
    rows_ = cols_ = 0;

    free(data_);
}


void Matrix::print() const {std::cout << *this;};


Matrix& Matrix::operator= (const Matrix& other) {
    if (this == &other) return *this;

    this->~Matrix();

    rows_ = other.rows_;
    cols_ = other.cols_;

    size_t bytes = sizeof(float) * rows_ * cols_;

    data_ = (float*)malloc(bytes);

    memcpy(data_, other.data_, bytes);

    return *this;
}


float& Matrix::operator() (size_t i, size_t j) {
    return data_[IDX2C(i,j,rows_)];
}


float Matrix::operator() (size_t i, size_t j) const {
    return data_[IDX2C(i,j,rows_)];
}


std::ostream &operator<<(std::ostream &out, const Matrix &m) {
    for (int j = 0; j < m.cols_; j++) {
        for (int i = 0; i < m.rows_; i++) {
            out << m.data_[IDX2C(i,j,m.rows_)] << ' ';
        }
        out << std::endl;
    }

    return out;
}


size_t Matrix::Rows() const {
    return rows_;
}


size_t Matrix::Cols() const {
    return cols_;
}

float *Matrix::Data() {
    return data_;
}

const float *Matrix::Data() const {
    return data_;
}

