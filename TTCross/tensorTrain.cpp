#include<numeric>
#include "tensorTrain.hpp"
#include "tensor.hpp"
#include "unfoldingMatrix.hpp"
#include "skeleton.hpp"

void TensorTrain::TTCross(const Tensor& t, double eps) {
    
    auto sizes = t.Sizes();
    size_t d =  t.Dimension();
    UnfoldingMatrix A(t);

    ttRanks.push_back(1);


    for (int k = 0; k < d; k++) {
        size_t n = ttRanks[k] * sizes[k], m = std::accumulate(sizes.begin() + k + 1, sizes.end(), 1, std::multiplies<size_t>());
        A = A.Reshape(n,m);

        auto [I,J] = Skeleton(A, eps);

        ttRanks.push_back(I.size());

        TMatrix U = A.ExplicitCols(J);
        TMatrix A_hat = A.ExplicitMaxvol(I,J);

        cores.push_back(Core(U * A_hat.Inverse(), ttRanks[k-1], sizes[k], ttRanks[k]));

        A.Compress(I);
    }
}