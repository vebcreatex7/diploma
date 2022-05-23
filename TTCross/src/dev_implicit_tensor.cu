#include "../include/dev_implicit_tensor.cuh"

DevImplicitTensor::DevImplicitTensor(size_t d, std::vector<size_t>  s, std::function<double(const std::vector<size_t> &)> f) :
        d_(d), s_(std::move(s)), f_(std::move(f)) {
    leftNestedSequence_.emplace_back(std::vector<size_t>());
    k_ = 0;
};

size_t DevImplicitTensor::Dimension() const {
    return d_;
}

const std::vector<size_t>& DevImplicitTensor::Sizes() const {
    return s_;
}


double DevImplicitTensor::operator()(size_t i, size_t j) const {
    std::vector<size_t> leftIdxs = unfoldRowIdx(i);
    std::vector<size_t> rightIdxs = unfoldColIdx(j);

    leftIdxs.insert(leftIdxs.end(), rightIdxs.begin(), rightIdxs.end());

    return f_(leftIdxs);
}

void DevImplicitTensor::Reshape(std::vector<size_t> I) {
    std::sort(I.begin(),I.end());
    std::vector<std::vector<size_t>> newLeftNestedSequence(I.size());

    for (size_t i = 0; i < I.size(); i++) {
        newLeftNestedSequence[i] = unfoldRowIdx(I[i]);
    }

    k_++;
    leftNestedSequence_ = newLeftNestedSequence;
}

std::vector<size_t> DevImplicitTensor::unfoldRowIdx(size_t p) const {
    std::vector<size_t> idxs = leftNestedSequence_[p / s_[k_]];
    idxs.push_back(p % s_[k_]);

    return idxs;
}

std::vector<size_t> DevImplicitTensor::unfoldColIdx(size_t p) const {
    int n = d_ - (k_ + 1);

    std::vector<size_t> idxs(n);

    size_t product = std::accumulate(s_.begin() + k_ + 1, s_.end(), 1, std::multiplies<>());

    for (size_t i = k_ + 1, j = 0; i < d_ - 1; i++, j++) {
        product /= s_[i];
        idxs[j] = p / product;
        p %= product;
    }

    idxs[n-1] = p;

    return idxs;
}

