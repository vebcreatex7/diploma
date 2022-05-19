#ifndef SPARSE_TENSOR_H
#define SPARSE_TENSOR_H

#include <vector>

#define seed_val 2

class Tensor {
    size_t d_;
    std::vector<size_t> sizes_;
    size_t overallSize_{};
    float* data_;

    public:
    Tensor() : d_(0), data_(nullptr) {};
    Tensor(size_t d, std::vector<size_t>  sizes);
    void FillSparse(float maxVal, float density);
    void FillSin();
    size_t OverallSize() const {return overallSize_;}
    double f(const std::vector<size_t>& idxs) const {return this->operator()(idxs);}


    double operator()(const std::vector<size_t>& idxs) const;
    double operator()(size_t p) const;
};

#endif