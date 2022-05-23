#ifndef DEV_IMPLICIT_TENSOR_CUH
#define DEV_IMPLICIT_TENSOR_CUH

#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>

class DevImplicitTensor {
private:
    size_t d_;
    std::vector<size_t> s_;
    size_t k_;
    std::function<double(const std::vector<size_t> &)> f_;
    std::vector<std::vector<size_t>> leftNestedSequence_;

    std::vector<size_t> unfoldRowIdx(size_t p) const;
    std::vector<size_t> unfoldColIdx(size_t p) const;

public:
    DevImplicitTensor(size_t d, std::vector<size_t>  n, std::function<double(const std::vector<size_t> &)> f);
    double operator()(size_t i, size_t j) const;
    size_t Dimension() const;
    const std::vector<size_t>&  Sizes() const;

    void Reshape(std::vector<size_t> I);
};

#endif //DEV_IMPLICIT_TENSOR_CUH
