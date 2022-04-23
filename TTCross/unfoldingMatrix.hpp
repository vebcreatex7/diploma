#include "tensor.hpp"
#include "matrix.hpp"

class UnfoldingMatrix : public Tensor {
private:
    size_t n, m;
    std::vector<std::vector<size_t>>  I;

public:
    UnfoldingMatrix(const Tensor& t, size_t n, size_t m);
    TMatrix ExplicitRows(const std::vector<size_t>& I);
    TMatrix ExplicitCols(const std::vector<size_t>& J);
    TMatrix ExplicitMaxvol(const std::vector<size_t>& I, const std::vector<size_t>& J);
};
