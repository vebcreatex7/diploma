#ifndef UNFOLDING_MATRIX_H
#define UNFOLDING_MATRIX_H

#include <numeric>

#include "implicit_tensor.cuh"
#include "matrix.cuh"
#include "host_matrix.cuh"
#include "dev_matrix.cuh"


class UnfoldingMatrix {
private:
    size_t rows_, cols_;
    ImplicitTensor* t_;

public:
    UnfoldingMatrix();
    UnfoldingMatrix(ImplicitTensor& t, size_t rows, size_t cols);

    size_t Rows() const {return rows_;}
    size_t Cols() const {return cols_;}

    TMatrix ExplicitRows(const std::vector<size_t>& I) const;
    TMatrix ExplicitCols(const std::vector<size_t>& J) const;
    TMatrix ExplicitMaxvol(const std::vector<size_t>& I,const std::vector<size_t>& J) const;

    Matrix GetRows(const std::vector<size_t>& I) const;
    Matrix GetCols(const std::vector<size_t>& J) const;
    Matrix GetMaxvol(const std::vector<size_t>& I, const std::vector<size_t>& J) const;

    
    void print() const;

    double operator() (size_t i, size_t j) const;
    UnfoldingMatrix& operator= (const UnfoldingMatrix& other);

    friend std::ostream& operator<< (std::ostream& os, const UnfoldingMatrix& m);

};

#endif