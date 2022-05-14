#include "../include/implicit_tensor.hpp"

ImplicitTensor::ImplicitTensor(size_t d, const std::vector<size_t>& s, std::function<double(const std::vector<size_t> &)> f) :
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

void ImplicitTensor::Reshape(std::vector<size_t> I) {
    std::sort(I.begin(),I.end());
    std::vector<std::vector<size_t>> newLeftNestedSequence(I.size());

    for (size_t i = 0; i < I.size(); i++) {
        newLeftNestedSequence[i] = unfoldRowIdx(I[i]);
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
    int n = d_ - (k_ + 1);

    std::vector<size_t> idxs(n);

    size_t product = std::accumulate(s_.begin() + k_ + 1, s_.end(), 1, std::multiplies<size_t>());

    for (size_t i = k_ + 1, j = 0; i < d_ - 1; i++, j++) {
        product /= s_[i];
        idxs[j] = p / product;
        p %= product;
    }

    idxs[n-1] = p;

    return idxs;
}




