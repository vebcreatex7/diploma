#ifndef UNFOLDING_MATRIX_H
#define UNFOLDING_MATRIX_H

#include <numeric>

#include "implicitTensor.hpp"
#include "matrix.hpp"


class UnfoldingMatrix {
private:
    size_t n_, m_;
    ImplicitTensor* t_;

public:
    UnfoldingMatrix();
    UnfoldingMatrix(ImplicitTensor& t, size_t n, size_t m);
    size_t N() const {return n_;}
    size_t M() const {return m_;}
    TMatrix ExplicitRows(const std::vector<size_t>& I) const;
    TMatrix ExplicitCols(const std::vector<size_t>& J) const;
    TMatrix ExplicitMaxvol(const std::vector<size_t>& I, const std::vector<size_t>& J) const;
    double operator() (size_t i, size_t j) const;
    UnfoldingMatrix& operator= (const UnfoldingMatrix& other);
};

#endif