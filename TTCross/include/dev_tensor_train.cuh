#ifndef DEV_TENSOR_TRAIN_CUH
#define DEV_TENSOR_TRAIN_CUH

#include <vector>
#include<numeric>

#include "dev_core.cuh"
#include "skeleton.cuh"
#include "implicit_tensor.cuh"
#include "unfolding_matrix.cuh"

class DevTensorTrain {
private:
    std::vector<DevCore> cores_;
    std::vector<size_t> ttRanks_;
    std::vector<size_t> sizes_;
    size_t foundMaxRank;

    double* devRes{};
    double* devTmp{};

public:

    ~DevTensorTrain();

    const std::vector<DevCore>& Cores() const;
    std::vector<size_t> TTRanks() const;

    void TTCross(ImplicitTensor t, double eps);
    void TTCross(ImplicitTensor t, size_t maxR, double eps);
    void TTCross(ImplicitTensor t, const std::vector<size_t>& upperBoundRanks, double eps);


    double operator()(const std::vector<size_t>& idxs);
    double operator()(size_t p);
};



#endif //DEV_TENSOR_TRAIN_CUH
