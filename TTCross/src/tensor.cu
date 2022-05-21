#include <numeric>
#include <random>
#include <utility>

#include "../include/tensor.cuh"

Tensor::Tensor(size_t d, std::vector<size_t> sizes) : d_(d), sizes_(std::move(sizes)) {
    overallSize_ = std::accumulate(sizes_.begin(), sizes_.end(), 1, std::multiplies<>());

    data_ = new double[overallSize_]();
}

void Tensor::FillSparse(double maxVal, double density) {
    std::mt19937 MyRNG;
    MyRNG.seed(seed_val);
    std::uniform_real_distribution<double> prob;
    std::uniform_real_distribution<double> val(0, maxVal);

    for (size_t i = 0; i < overallSize_; i++) {
        if (prob(MyRNG) <= density) {
            data_[i] = val(MyRNG);
        } else data_[i] = 0.;
    }
}

void Tensor::FillSin() {
    for (size_t i = 0; i < overallSize_; i++) {
        data_[i] = (double)sin(i);
    }
}

double Tensor::operator()(const std::vector<size_t>& idxs) const {
        size_t remainder = overallSize_;
        size_t p = 0;

        for (size_t i = 0; i < idxs.size() - 1; i++) {
            remainder /= sizes_[i];
            p += idxs[i] * remainder;
        }

        p += idxs.back();

        return data_[p];
}

double Tensor::operator()(size_t p) const {
    return data_[p];
}