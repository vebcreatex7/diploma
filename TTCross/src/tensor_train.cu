#include "../include/tensor_train.cuh"
#include "../include/dev_matrix.cuh"

void TensorTrain::TTCross(ImplicitTensor t, double eps) {
    std::vector<size_t> upperBoundRanks(t.Dimension()-1, upperBoundRank);

    return TTCross(t, upperBoundRanks, eps);
}

void TensorTrain::TTCross(ImplicitTensor t, size_t maxRank, double eps) {
    std::vector<size_t> upperBoundRanks(t.Dimension(), maxRank);

    return TTCross(t,upperBoundRanks, eps);
}

void TensorTrain::TTCross(ImplicitTensor t, const std::vector<size_t>& upperBoundRanks, double eps) {
    sizes_ = t.Sizes();
    size_t d =  t.Dimension();
    
    UnfoldingMatrix A;
    std::vector<size_t> I, J;

    ttRanks_.resize(d + 1);
    cores_.resize(d);

    ttRanks_[0] = 1;

    for (size_t k = 0; k < d-1; k++) {
        size_t n = ttRanks_[k] * sizes_[k], m = std::accumulate(sizes_.begin() + k + 1, sizes_.end(), 1, std::multiplies<>());

        A = UnfoldingMatrix(t, n, m);

        auto idxsMax = Skeleton(A, upperBoundRanks[k], eps);

        I = idxsMax.first;
        J = idxsMax.second;

        ttRanks_[k+1] = I.size();

        DevMatrix<float> devU(A.GetCols(J));
        DevMatrix<float> devAHat(A.GetMaxvol(I,J));
        DevMatrix<float> devAHatInv = devAHat.Inverse();
        DevMatrix<float> devTmp = devU * devAHatInv;

        TMatrix U = A.ExplicitCols(J);
        U = devU.ToTMatrix();

        TMatrix A_hat = A.ExplicitMaxvol(I,J);
        A_hat = devAHat.ToTMatrix();

        TMatrix A_hat_inv = A_hat.Inverse();
        A_hat_inv = devAHatInv.ToTMatrix();

        TMatrix tmp = U * A_hat_inv;
        //TMatrix tmp = devTmp.ToTMatrix();

        cores_[k] = Core(tmp, ttRanks_[k], sizes_[k], ttRanks_[k+1]);
        cores_[k].SetMatrices(devTmp, ttRanks_[k], sizes_[k], ttRanks_[k+1]);

        if (k != d-2) t.Reshape(I);
    }

    ttRanks_[d] = 1;

    TMatrix R = A.ExplicitRows(I);
    cores_[d-1] = Core(R, ttRanks_[d-1], sizes_[d-1], ttRanks_[d]);

    DevMatrix<float> devR = DevMatrix<float>(A.GetRows(I));
    cores_[d-1].SetMatrices(devR, ttRanks_[d-1], sizes_[d-1], ttRanks_[d]);

}

const std::vector<Core>& TensorTrain::Cores() const {
    return cores_;
}

std::vector<size_t> TensorTrain::TTRanks() const {
    return ttRanks_;
}

size_t TensorTrain::OverallSize() const {
    size_t overallSize = 0;
    for (size_t i = 0; i < sizes_.size(); i++) {
        overallSize += ttRanks_[i]*sizes_[i]*ttRanks_[i+1];
    }

    return overallSize;
}


float TensorTrain::value(const vector<size_t> &idxs) const {
    DevMatrix<float> devRes = cores_[0].Matrix(idxs[0]);

    for (size_t i = 1; i < idxs.size(); i++) {
        DevMatrix<float> devTmp = cores_[i].Matrix(idxs[i]);

        devRes = devRes * devTmp;
    }

    return devRes.ToHost()(0,0);
}

float TensorTrain::linearValue(size_t p) const {
    size_t d = sizes_.size();
    std::vector<size_t> idxs(d);
    size_t product = std::accumulate(sizes_.begin(), sizes_.end(), 1, std::multiplies<>());

    for (size_t i = 0; i < d - 1; i++) {
        product /= sizes_[i];
        idxs[i] = p / product;
        p %= product;
    }

    idxs[d - 1] = p;

    return value(idxs);
}

float TensorTrain::operator()(const std::vector<size_t>& idxs) const {
    TMatrix res = cores_[0](idxs[0]);

    for (size_t i = 1; i < idxs.size(); i++) {
        TMatrix tmp = cores_[i](idxs[i]);
        
        res = res * tmp;
    }

    return res[0][0];
}



float TensorTrain::operator()(size_t p) const {
    size_t d = sizes_.size();
    std::vector<size_t> idxs(d);
    size_t product = std::accumulate(sizes_.begin(), sizes_.end(), 1, std::multiplies<>());

    for (size_t i = 0; i < d - 1; i++) {
        product /= sizes_[i];
        idxs[i] = p / product;
        p %= product;
    }

    idxs[d - 1] = p;

    return this->operator()(idxs);
}
