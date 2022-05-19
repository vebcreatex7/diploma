#ifndef HOST_MATRIX_H
#define HOST_MATRIX_H

#define IDX2C(i,j,rows) (((j)*(rows)) + (i))

#include <cstdlib>
#include <cstring>
#include <iostream>


class Matrix {
private:
    size_t rows_, cols_;
    float* data_;

public:
    Matrix();
    Matrix(size_t rows, size_t cols, float val = 0);
    Matrix(const Matrix& other);
    ~Matrix();

    size_t Rows() const;
    size_t Cols() const;
    float* Data();
    const float* Data() const;
    void print() const;

    Matrix& operator= (const Matrix& other);
    float& operator() (size_t i, size_t j);
    float operator() (size_t i, size_t j) const;

    friend std::ostream& operator << (std::ostream& out, const Matrix& m);

};

#endif
