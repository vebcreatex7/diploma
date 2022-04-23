#include<numeric>
#include "tensorTrain.hpp"
#include "tensor.hpp"
#include "unfoldingMatrix.hpp"
#include "skeleton.hpp"

void TensorTrain::TTCross(const Tensor& t, double eps) {
    ttRanks.push_back(1);
    auto sizes = t.Sizes();
    size_t n = sizes[0], m = std::accumulate(++sizes.begin(), sizes.end(), 1, std::multiplies<int>());

    size_t d =  t.Dimension();

    ttRanks.push_back(1);

    UnfoldingMatrix A(t, n, m);


    for (int k = 1; k < d; k++) {
        auto [I,J] = Skeleton(A, eps);

        ttRanks.push_back(I.size());

        TMatrix U = A.ExplicitCols(J);
        TMatrix A_hat = A.ExplicitMaxvol(I,J);

        cores.push_back(Core(U * A_hat.Inverse(), ttRanks[k-1], sizes[k], ttRanks[k]));
    }
}