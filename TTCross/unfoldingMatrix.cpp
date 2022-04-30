#include <numeric>

#include "unfoldingMatrix.hpp"
#include "tensor.hpp"


UnfoldingMatrix::UnfoldingMatrix(const ImplicitTensor& t, size_t n, size_t m) :
    n_(n), m_(m), t_(t) {}


double UnfoldingMatrix::operator()(size_t i, size_t j) const {
    return t_(i,j);
}

TMatrix UnfoldingMatrix::ExplicitRows(const std::vector<size_t>& I) const {
    size_t n = I.size();
    TMatrix res(n,m_,0.);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m_; j++) {
            res[i][j] = t_(I[i],j);
        }
    }

    return res;
}

TMatrix UnfoldingMatrix::ExplicitCols(const std::vector<size_t>& J) const {
    size_t m = J.size();
    TMatrix res(n_,m,0.);

    for (size_t i = 0; i < n_; i++) {
        for (int j = 0; j < m; j++) {
            res[i][j] = t_(i, J[j]);
        }
    }

    return res;
}

TMatrix UnfoldingMatrix::ExplicitMaxvol(const std::vector<size_t>& I, const std::vector<size_t>& J) const {
    size_t n = I.size();
    size_t m = J.size();
    TMatrix res(n,m,0.);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            res[i][j] = t_(I[i], J[j]);
        }
    }

    return res;
}
