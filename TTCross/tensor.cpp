#include "tensor.hpp"

Tensor::Tensor(size_t d, const std::vector<size_t>& n, const std::function<double(const std::vector<size_t> &)> f) : d_(d), n_(n), f_(f) {};

double Tensor::Get(const std::vector<size_t>& idxs) const {
    return f_(idxs);
}

size_t Tensor::Dimension() const {
    return d_;
}

const std::vector<size_t>& Tensor::Sizes() const {
    return n_;
}




