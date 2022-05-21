#ifndef HOST_MATRIX_H
#define HOST_MATRIX_H

#define IDX2C(i,j,rows) (((j)*(rows)) + (i))

#include <cstdlib>
#include <cstring>
#include <iostream>


class Matrix {
private:
    size_t rows_, cols_;
    double* data_;

public:
    Matrix();
    Matrix(size_t rows, size_t cols, double val = 0);
    Matrix(const Matrix& other);
    ~Matrix();

    size_t Rows() const;
    size_t Cols() const;
    double* Data();
    const double* Data() const;
    void print() const;

    Matrix& operator= (const Matrix& other);
    double& operator() (size_t i, size_t j);
    double operator() (size_t i, size_t j) const;

    friend std::ostream& operator << (std::ostream& out, const Matrix& m);

};

#endif
