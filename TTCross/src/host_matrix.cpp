#include "../include/host_matrix.hpp"

template <class T>
Matrix<T>::Matrix() : rows_(0), cols_(0), data_(NULL) {}

template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols, T val) : rows_(rows), cols_(cols) {
    size_t bytes = sizeof(T) * rows_ * cols_;

    data_ = (T*)malloc(bytes);
}

template <class T>
Matrix<T>::Matrix(const Matrix<T>& other) : rows_(other.rows_), cols_(other.cols_) {
    size_t bytes = sizeof(T) * rows_ * cols_;

    std::memcpy(data_, other.data_, bytes);
}

template <class T>
Matrix<T>::~Matrix() {
    rows_ = cols_ = 0;

    free(data_);
}

template <class T>
Matrix<T>& Matrix<T>::operator= (const Matrix<T>& other) {
    this->~Matrix();

    rows_ = other.rows_;
    cols_ = other.cols_;

    size_t bytes = sizeof(T) * rows_ * cols_;

    data_ = (T*)malloc(bytes);

    std::memcpy(data_, other.data_, bytes);

    return *this; 
}

template <class T>
T& Matrix<T>::operator() (size_t i, size_t j) {
    return data_[IDX2C(i,j,rows_)];
}

template <class T>
T Matrix<T>::operator() (size_t i, size_t j) const {
    return data_[IDX2C(i,j,rows_)];
}


