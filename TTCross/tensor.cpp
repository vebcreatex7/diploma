#include "tensor.hpp"

ImplicitTensor::ImplicitTensor(size_t d, const std::vector<size_t>& s, const std::function<double(const std::vector<size_t> &)> f) :
    d_(d), s_(s), f_(f) {
        leftNestedSequence_.push_back(std::vector<size_t>());
        k_ = 0;
    };

size_t ImplicitTensor::Dimension() const {
    return d_;
}

const std::vector<size_t>& ImplicitTensor::Sizes() const {
    return s_;
}

double ImplicitTensor::operator()(size_t i, size_t j) const {
    std::vector<size_t> leftIdxs = unfoldRowIdx(i);
    std::vector<size_t> rightIdxs = unfoldColIdx(j);

    leftIdxs.insert(leftIdxs.end(), rightIdxs.begin(), rightIdxs.end());

    return f_(leftIdxs);
}

void ImplicitTensor::Reshape(const std::vector<size_t>& I) {
    std::vector<std::vector<size_t>> newLeftNestedSequence(I.size());

    for (int i = 0; i < I.size(); i++) {
        newLeftNestedSequence.push_back(unfoldRowIdx(I[i]));
    }

    k_++;
    leftNestedSequence_ = newLeftNestedSequence;
}

std::vector<size_t> ImplicitTensor::unfoldRowIdx(size_t p) const {
    std::vector<size_t> idxs = leftNestedSequence_[p / s_[k_]];
    idxs.push_back(p % s_[k_]);

    return idxs;
}

std::vector<size_t> ImplicitTensor::unfoldColIdx(size_t p) const {
    int n = d_ - k_ + 1;

    std::vector<size_t> idxs(n);

    size_t product = std::accumulate(s_.begin(), s_.end(), 1, std::multiplies<size_t>());

    for (int i = k_ + 1, j = 0; i < d_; i++, j++) {
        product /= s_[i];
        idxs[j] = p / product;
        p %= product;
    }

    return idxs;
}




