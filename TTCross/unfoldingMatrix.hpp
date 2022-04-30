#include "tensor.hpp"
#include "matrix.hpp"

class UnfoldingMatrix {
private:
    size_t n_, m_;
    const ImplicitTensor& t_;

public:
    UnfoldingMatrix(const ImplicitTensor& t, size_t n, size_t m);
    UnfoldingMatrix(const UnfoldingMatrix& t);
    TMatrix ExplicitRows(const std::vector<size_t>& I) const;
    TMatrix ExplicitCols(const std::vector<size_t>& J) const;
    TMatrix ExplicitMaxvol(const std::vector<size_t>& I, const std::vector<size_t>& J) const;
    double operator() (size_t i, size_t j) const;
};
