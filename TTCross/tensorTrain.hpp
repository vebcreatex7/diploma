#ifndef TENSOR_TRAIN_H
#define TENSOR_TRAIN_H

#include <vector>
#include<numeric>

#include "implicitTensor.hpp"
#include "unfoldingMatrix.hpp"
#include "skeleton.hpp"
#include "core.hpp"

class TensorTrain {
private:
    std::vector<Core> cores;
    std::vector<size_t> ttRanks;

public:
    const std::vector<Core>& Cores() const;
    std::vector<size_t> TTRanks() const;
    void TTCross(ImplicitTensor t, double eps);
    double operator()(const std::vector<size_t>& idxs) const;
};

#endif