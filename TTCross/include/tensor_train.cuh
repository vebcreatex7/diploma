#ifndef TENSOR_TRAIN_H
#define TENSOR_TRAIN_H

#include <vector>
#include<numeric>

#include "implicit_tensor.cuh"
#include "unfolding_matrix.cuh"
#include "skeleton.cuh"
#include "core.cuh"

const int upperBoundRank = 100000;

class TensorTrain {
private:
    std::vector<Core> cores_;
    std::vector<size_t> ttRanks_;
    std::vector<size_t> sizes_;


public:
    const std::vector<Core>& Cores() const;
    std::vector<size_t> TTRanks() const;
    void TTCross(ImplicitTensor t, double eps);
    void TTCross(ImplicitTensor t, size_t maxR, double eps);
    void TTCross(ImplicitTensor t, const std::vector<size_t>& upperBoundRanks, double eps);
    size_t OverallSize() const;

    float value(const std::vector<size_t>& idxs) const;
    float linearValue(size_t p) const;

    float operator()(const std::vector<size_t>& idxs) const;
    float operator()(size_t p) const;
};

#endif