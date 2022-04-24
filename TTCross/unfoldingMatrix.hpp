#include "tensor.hpp"
#include "matrix.hpp"

class UnfoldingMatrix : public Tensor {
private:
    size_t n_, m_;
    size_t k_;
    std::vector<std::vector<size_t>>  I;

    std::vector<size_t> row(size_t p);
    std::vector<size_t> col(size_t p);

public:
    UnfoldingMatrix(const Tensor& t);
    UnfoldingMatrix(const UnfoldingMatrix& t);
    UnfoldingMatrix Reshape(size_t n, size_t m);
    UnfoldingMatrix Compress(const std::vector<size_t>& I);
    TMatrix ExplicitRows(const std::vector<size_t>& I);
    TMatrix ExplicitCols(const std::vector<size_t>& J);
    TMatrix ExplicitMaxvol(const std::vector<size_t>& I, const std::vector<size_t>& J);
    double Get(size_t i, size_t j) const;
};
