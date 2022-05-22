#include <random>
#include <vector>
#include <ctime>

#include "include/tensor_train.cuh"
#include "include/tensor.cuh"
#include "include/dev_tensor_train.cuh"


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

    clock_t begin = clock();

    tt.TTCross(impTensor, maxR, 0.1);

    clock_t end = clock();
    std::cout << "CPU tt time = " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    double norm = 0., ttNorm = 0., devTtNorm = 0.;
    begin = clock();
    for (size_t i = 0; i < tensor.OverallSize(); i++) {
        double val = tensor(i);
        double ttVal = tt(i);

        norm += (double)val * val;
        ttNorm += (double)ttVal * ttVal;
    }
    end = clock();
    std::cout << "CPU calculate time = " << double(end - begin) / CLOCKS_PER_SEC << std::endl;


    DevTensorTrain devTt;

    begin = clock();

    devTt.TTCross(impTensor, maxR, 0.1);

    cudaDeviceSynchronize();
    end = clock();
    std::cout << "GPU tt time = " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    begin = clock();
    for (size_t i = 0; i < tensor.OverallSize(); i++) {
        double devTtVal = devTt(i);

        devTtNorm += (double)devTtVal * devTtVal;
    }
    end = clock();
    std::cout << "GPU calculate time = " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::cout << std::setprecision(5) << std::fixed;
    std::cout
            << "norm = " << sqrt(norm) << std::endl
            << "ttNorm = " << sqrt(ttNorm) << std::endl
            << "devTtNorm = " << sqrt(devTtNorm) << std::endl;
    /*
    begin = clock();
    tt.devTTCross(impTensor, maxR, 0.1);
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "GPU tt time = " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    begin = clock();
    for (size_t i = 0; i < tensor.OverallSize(); i++) {
        double devTtVal = tt.linearValue(i);

        devTtNorm += devTtVal * devTtVal;
    }
    cudaDeviceSynchronize();
    end = clock();
    std::cout << "GPU calculate time = " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

    std::cout << std::setprecision(5) << std::fixed;
    std::cout
            << "norm = " << sqrt(norm) << std::endl
            << "ttNorm = " << sqrt(ttNorm) << std::endl
            << "devTtNorm = " << sqrt(devTtNorm) << std::endl;
    */
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