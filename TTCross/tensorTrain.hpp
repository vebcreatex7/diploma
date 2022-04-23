#include <vector>
#include "tensor.hpp"
#include "core.hpp"

class TensorTrain {
private:
    std::vector<Core> cores;
    std::vector<size_t> ttRanks;

public:
    TensorTrain();
    std::vector<Core>& Cores() const;
    void TTCross(const Tensor& t, double eps);
};

