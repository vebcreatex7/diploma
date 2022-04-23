#include <vector>
#include <functional>

class Tensor {
private:
    size_t d_;
    std::vector<size_t> n_;
    std::function<double(const std::vector<size_t> &)> f_;
public:
    Tensor(size_t d, const std::vector<size_t>& n, const std::function<double(const std::vector<size_t> &)> f);
    double Get(std::vector<size_t> const& idxs) const;
    size_t Dimension() const;
    const std::vector<size_t>&  Sizes() const;
};


