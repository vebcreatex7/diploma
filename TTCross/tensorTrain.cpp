#include<numeric>
#include "tensorTrain.hpp"
#include "tensor.hpp"
#include "unfoldingMatrix.hpp"
#include "skeleton.hpp"

void TensorTrain::TTCross(ImplicitTensor t, double eps) {
    auto sizes = t.Sizes();
    size_t d =  t.Dimension();

    ttRanks.push_back(1);


    for (int k = 1; k < d; k++) {
        size_t n = ttRanks[k-1] * sizes[k], m = std::accumulate(sizes.begin() + k + 1, sizes.end(), 1, std::multiplies<size_t>());

        UnfoldingMatrix A(t, n, m);

        auto [I,J] = Skeleton(A, eps);

        ttRanks.push_back(I.size());

        TMatrix U = A.ExplicitCols(J);
        TMatrix A_hat = A.ExplicitMaxvol(I,J);

        cores.push_back(Core(U * A_hat.Inverse(), ttRanks[k-1], sizes[k], ttRanks[k]));

        t.Reshape(I);
    }
}