#include <numeric>

#include "unfoldingMatrix.hpp"
#include "tensor.hpp"

UnfoldingMatrix::UnfoldingMatrix(const Tensor& t) : Tensor(t) {}

UnfoldingMatrix::UnfoldingMatrix(const UnfoldingMatrix& t) : Tensor(t) {}

UnfoldingMatrix UnfoldingMatrix::Reshape(size_t n, size_t m) {
    TMatrix res();
}

std::vector<size_t> UnfoldingMatrix::row(size_t p) {
    std::vector<size_t> idxs;

    if (k_ == 0) {
        idxs.push_back(p);
        return idxs;
    }

    idxs.insert(idxs.end(), I[p % I.size()].begin(), I[p % I.size()].end());
    idxs.push_back(p / I.size());

    return idxs;
}

std::vector<size_t> UnfoldingMatrix::col(size_t p) {

    std::vector<size_t> idxs;

    std::vector<size_t> n;
    n.assign(Sizes().begin() + k_ + 1, Sizes().end());

    size_t product = std::accumulate(n.begin(), n.end(), 1, std::multiplies<size_t>());

    for (int i = n.size() - 1; i >= 0; i--) {
        product /= n[i];
        size_t idx = p / product;
        p %= product;

        idxs.push_back(idx);
    }

    return idxs;
}

double UnfoldingMatrix::Get(size_t i, size_t j) const {
    vector<size_t> idxs;


    
}

