#ifndef IMPLICIT_TENSOR_H
#define IMPLICIT_TENSOR_H


#include <vector>
#include <functional>
#include <numeric>

class ImplicitTensor {
private:
    size_t d_;
    std::vector<size_t> s_;
    size_t k_;
    std::function<double(const std::vector<size_t> &)> f_;
    std::vector<std::vector<size_t>> leftNestedSequence_;

    std::vector<size_t> unfoldRowIdx(size_t p) const;
    std::vector<size_t> unfoldColIdx(size_t p) const;

public:
    ImplicitTensor(size_t d, const std::vector<size_t>& n, std::function<double(const std::vector<size_t> &)> f);
    double operator()(size_t i, size_t j) const;
    size_t Dimension() const;
    const std::vector<size_t>&  Sizes() const;

    void Reshape(const std::vector<size_t>& I);
};


#endif