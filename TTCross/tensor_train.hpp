#ifndef TENSOR_TRAIN_H
#define TENSOR_TRAIN_H

#include <vector>
#include<numeric>

#include "implicit_tensor.hpp"
#include "unfolding_matrix.hpp"
#include "skeleton.hpp"
#include "core.hpp"

class TensorTrain {
private:
    std::vector<Core> cores_;
    std::vector<size_t> ttRanks_;
    std::vector<size_t> sizes_;

public:
    const std::vector<Core>& Cores() const;
    std::vector<size_t> TTRanks() const;
    void TTCross(ImplicitTensor t, size_t maxR, double eps);
    double operator()(const std::vector<size_t>& idxs) const;
    double operator()(size_t p) const;
};

#endif