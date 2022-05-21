#include <random>
#include <vector>
#include <ctime>

#include "include/tensor_train.cuh"
#include "include/tensor.cuh"


using namespace std;


TMatrix SparseMatrix(int n, int m, double density) {
    TMatrix res(n,m,0.);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            double val = rand() / double(RAND_MAX);
            if (val <= density) {
                res[i][j] = sin(i + j);
            }
        }

    return res;
}

void TT() {
    std::vector<size_t> sizes{10,10,10};

    Tensor expTensor(sizes.size(), sizes);
    expTensor.FillSparse(1, 0.01);
    //expTensor.FillSin();

    ImplicitTensor impTensor(sizes.size(), sizes, std::bind(&Tensor::f, expTensor,std::placeholders::_1));

    std::vector<size_t> upperBoundRanks = {2, 2, 2, 12, 12, 3, 2, 2};

    TensorTrain tt;

    clock_t begin = clock();
    tt.TTCross(impTensor,10, 0.0001);
    clock_t end = clock();
    std::cout << "elapsed time = " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
    std::vector<size_t> ttRanks = tt.TTRanks();
    std::cout << "ttRanks = ";
    for (auto& r : ttRanks) std::cout << r << ' ';
    std::cout << std::endl;

    double maxDiff = 0;
    double expNorm = 0, ttNorm = 0, diffNorm, devNorm = 0.;
    size_t overallSize = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<>());
    for (size_t i = 0; i < overallSize; i++) {
        double expVal = expTensor(i);
        double ttVal = tt(i);
        double devVal = tt.linearValue(i);

        expNorm += expVal * expVal;
        ttNorm += ttVal * ttVal;
        diffNorm += (expVal - ttVal) * (expVal - ttVal);
        devNorm += devVal * devVal;
        double diff = abs(expVal - ttVal);

        if (diff > maxDiff) maxDiff = diff;
    }

    expNorm = sqrt(expNorm);
    ttNorm = sqrt(ttNorm);
    diffNorm = sqrt(diffNorm);

    std::cout << std::setprecision(5) << std::fixed;
    std::cout
            << "overall size of original tensor " << expTensor.OverallSize() << std::endl
            << "overall size of tensor train " << tt.OverallSize() << std::endl
            << "explicit norm = " << expNorm << std::endl
            << "tt norm = " << ttNorm << std::endl
             << "dev norm = " << devNorm << std::endl
            << "delta norm = " << diffNorm << std::endl;


}

void CPUvsGPU() {
    size_t d;
    std::cin >> d;

    std::vector<size_t> sizes(d);
    for (auto& a : sizes)
            std::cin >> a;

    size_t maxR;
    std::cin >> maxR;

    double density;
    std::cin >> density;

    Tensor tensor(d,sizes);
    tensor.FillSparse(1, density);

    ImplicitTensor impTensor(sizes.size(), sizes, std::bind(&Tensor::f, tensor,std::placeholders::_1));

    TensorTrain tt;

    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    tt.TTCross(impTensor, maxR, 0.1);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "CPU TTCross time =" << time << std::endl;

    cudaEventRecord(start, 0);

    tt.devTTCross(impTensor, maxR, 0.1);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "GPU TTCross time =" << time << std::endl;

    double norm = 0., ttNorm = 0., devTtNorm = 0.;

    cudaEventRecord(start, 0);
    for (size_t i = 0; i < tensor.OverallSize(); i++) {
        double val = tensor(i);
        double ttVal = tt(i);

        norm += (double)val * val;
        ttNorm += (double)ttVal * ttVal;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "CPU calculate time = " << time << std::endl;



    cudaEventRecord(start, 0);
    for (size_t i = 0; i < tensor.OverallSize(); i++) {
        double devTtVal = tt.linearValue(i);

        devTtNorm += devTtVal * devTtVal;
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "GPU calculate time = " << time << std::endl;

    std::cout << std::setprecision(5) << std::fixed;
    std::cout
            << "norm = " << norm << std::endl
            << "ttNorm = " << ttNorm << std::endl
            << "devTtNorm = " << devTtNorm << std::endl;

}



void MatricesMul() {
    size_t m,n,k;
    std::cin >> m >> n >> k;

    DevMatrix<double> A(m,k);
    DevMatrix<double> B(k,n);

    std::cin >> A;
    std::cin >> B;

    std::cout << A << std::endl;
    std::cout << B << std::endl;

    DevMatrix<double> C = A * B;

    std::cout << C;
}

void MatricesInv() {
    size_t n;
    std::cin >> n;

    DevMatrix<double> A = Random<double>(n,n, 0.1);

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    DevMatrix<double> InvA = A.Inverse();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "inv time = " << time << std::endl;

    cudaEventRecord(start, 0);

    DevMatrix<double> I = A * InvA;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    std::cout << "mul time = " << time << std::endl;
}

int main() {
    cudaFree(0);
    //MatricesMul();
    CPUvsGPU();
}