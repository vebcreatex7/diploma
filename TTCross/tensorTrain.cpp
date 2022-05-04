#include "tensorTrain.hpp"

void TensorTrain::TTCross(ImplicitTensor t, double eps) {
    auto sizes = t.Sizes();
    size_t d =  t.Dimension();
    
    UnfoldingMatrix A;
    std::vector<size_t> I, J;

    ttRanks.resize(d + 1);
    cores.resize(d);

    ttRanks[0] = 1;



    for (size_t k = 0; k < d-1; k++) {
        size_t n = ttRanks[k] * sizes[k], m = std::accumulate(sizes.begin() + k + 1, sizes.end(), 1, std::multiplies<size_t>());

        A = UnfoldingMatrix(t, n, m);

        auto idxs = Skeleton(A, eps);
        I = std::get<0>(idxs);
        J = std::get<1>(idxs);

        ttRanks[k+1] = I.size();

        TMatrix U = A.ExplicitCols(J);
        TMatrix A_hat = A.ExplicitMaxvol(I,J);

        TMatrix A_hat_inv = A_hat.Inverse();
        TMatrix tmp = U * A_hat_inv;

        cores[k] = Core(tmp, ttRanks[k], sizes[k], ttRanks[k+1]);

        if (k != d-2) t.Reshape(I);
    }

    ttRanks[d] = 1;

    TMatrix R = A.ExplicitRows(I);

    cores[d-1] = Core(R, ttRanks[d-1], sizes[d-1], ttRanks[d]);
}

const std::vector<Core>& TensorTrain::Cores() const {
    return cores;
}

std::vector<size_t> TensorTrain::TTRanks() const {
    return ttRanks;
}

double TensorTrain::operator()(const std::vector<size_t>& idxs) const {
    TMatrix res = cores[0](idxs[0]);

    for (size_t i = 1; i < idxs.size(); i++) {
        TMatrix tmp = cores[i](idxs[i]);
        
        res = res * tmp;
    }

    return res[0][0];
}