#include "../include/dev_tensor_train.cuh"
#include "../include/dev_matrix.cuh"

const std::vector<DevCore> &DevTensorTrain::Cores() const {
    return cores_;
}

std::vector<size_t> DevTensorTrain::TTRanks() const {
    return ttRanks_;
}

void DevTensorTrain::TTCross(ImplicitTensor t, double eps) {
    std::vector<size_t> upperBoundRanks(t.Dimension()-1, 10000);

    return TTCross(t, upperBoundRanks, eps);
}

void DevTensorTrain::TTCross(ImplicitTensor t, size_t maxR, double eps) {
    std::vector<size_t> upperBoundRanks(t.Dimension(), maxR);

    return TTCross(t,upperBoundRanks, eps);
}

void DevTensorTrain::TTCross(ImplicitTensor t, const vector<size_t> &upperBoundRanks, double eps) {
    sizes_ = t.Sizes();
    size_t d =  t.Dimension();
    foundMaxRank = 0;

    UnfoldingMatrix A;
    std::vector<size_t> I, J;

    ttRanks_.resize(d + 1);

    ttRanks_[0] = 1;

    for (size_t k = 0; k < d-1; k++) {
        size_t n = ttRanks_[k] * sizes_[k], m = std::accumulate(sizes_.begin() + k + 1, sizes_.end(), 1,
                                                                std::multiplies<>());

        A = UnfoldingMatrix(t, n, m);

        auto idxsMax = Skeleton(A, upperBoundRanks[k], eps);

        I = idxsMax.first;
        J = idxsMax.second;

        ttRanks_[k + 1] = I.size();
        foundMaxRank = ttRanks_[k + 1] > foundMaxRank ? ttRanks_[k + 1] : foundMaxRank;

        DevMatrix<double> devU(A.GetCols(J));
        DevMatrix<double> devAHatInv = DevMatrix<double>(A.GetMaxvol(I, J)).Inverse();
        DevMatrix<double> tmp = devU * devAHatInv;

        cores_.push_back(DevCore(tmp.Data(),ttRanks_[k], sizes_[k], ttRanks_[k+1]));

        if (k != d-2) t.Reshape(I);
    }

    ttRanks_[d] = 1;

    DevMatrix<double> R = DevMatrix<double>(A.GetRows(I));

    cores_.push_back(DevCore(R.Data(),ttRanks_[d-1], sizes_[d-1], ttRanks_[d]));

    size_t bytes = sizeof(double) * foundMaxRank;

    CudaErrHandler(cudaMalloc((void**)&devRes, bytes));
    CudaErrHandler(cudaMalloc((void**)&devTmp, bytes));


}


void plint(double* devData, size_t m, size_t n) {
    size_t bytes = sizeof(double) * m * n;

    auto* data = (double*)malloc(bytes);

    cudaMemcpy(data,devData, bytes, cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            std::cout << data[IDX2C(i,j,m)] << ' ';
        }
        std::cout << '\n';
    }

    free(data);
}

double DevTensorTrain::operator()(const vector<size_t> &idxs) {
    CudaErrHandler(cudaMemcpy(devRes,cores_[0].Matrix(idxs[0]), sizeof(double) * cores_[0].GetC(), cudaMemcpyDeviceToDevice));

    for (size_t i = 1; i < idxs.size(); i++) {
        //cudaMemset(tmp, 0, bytes);

        size_t m = 1, n = cores_[i].GetA(), k = cores_[i].GetC();

        double *core = cores_[i].Matrix(idxs[i]);

        MatrixMul(devRes, core, devTmp, m, n, k);

        double* swap = devRes;
        devRes = devTmp;
        devTmp = swap;
    }

    double val;
    CudaErrHandler(cudaMemcpy(&val, devRes, sizeof(double), cudaMemcpyDeviceToHost));

    return val;
}

double DevTensorTrain::operator()(size_t p) {
    size_t d = sizes_.size();
    std::vector<size_t> idxs(d);
    size_t product = std::accumulate(sizes_.begin(), sizes_.end(), 1, std::multiplies<>());

    for (size_t i = 0; i < d - 1; i++) {
        product /= sizes_[i];
        idxs[i] = p / product;
        p %= product;
    }

    idxs[d - 1] = p;

    return operator()(idxs);
}

DevTensorTrain::~DevTensorTrain() {
    CudaErrHandler(cudaFree(devRes));
    CudaErrHandler(cudaFree(devTmp));
}
