#ifndef HOST_MATRIX_H
#define HOST_MATRIX_H

#define IDX2C(i,j,rows) (((j)*(rows)) + (i))

#include <stdlib.h>
#include <cstring>

template <class T>
class Matrix {
    size_t rows_, cols_;
    T* data_;

    Matrix();
    Matrix(size_t rows, size_t cols, T val = 0);
    Matrix(const Matrix& other);
    ~Matrix();

    size_t Rows() const {return rows_;};
    size_t Cols() const {return cols_;};

    Matrix& operator= (const Matrix& other);
    T& operator() (size_t i, size_t j);
    T operator() (size_t i, size_t j) const;
};

#endif
