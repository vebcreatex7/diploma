#include <random>
#include <vector>
#include <limits>
#include <ctime>

#include "tensor_train.hpp"
#include "tensor.hpp"


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

int main() {  
    std::vector<size_t> sizes{10,10,10,10};

    Tensor expTensor(sizes.size(), sizes);
    expTensor.FillSparse(1, 1);

    ImplicitTensor impTensor(sizes.size(), sizes, std::bind(&Tensor::f, expTensor,std::placeholders::_1));

    TensorTrain tt;

    clock_t begin = clock();
    tt.TTCross(impTensor,50, 0.0001);
    clock_t end = clock();

    std::vector<size_t> ttRanks = tt.TTRanks();

    for (auto& r : ttRanks) std::cout << r << ' ';
    std::cout << std::endl;

    double maxDiff = 0;
    double expNorm = 0, ttNorm = 0, diffNorm;
    size_t overallSize = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<size_t>());
    for (size_t i = 0; i < overallSize; i++) {
        double expVal = expTensor(i);
        double ttVal = tt(i);

        expNorm += expVal * expVal;
        ttNorm += ttVal * ttVal;
        diffNorm += (expVal - ttVal) * (expVal - ttVal);

        double diff = abs(expVal - ttVal);

        if (diff > maxDiff) maxDiff = diff;
    }

    expNorm = sqrt(expNorm);
    ttNorm = sqrt(ttNorm);
    diffNorm = sqrt(diffNorm);

    std::cout << std::setprecision(5) << std::fixed;
    std::cout 
        << "explicit norm = " << expNorm << std::endl
        << "tt norm = " << ttNorm << std::endl
        << "delta norm = " << diffNorm << std::endl
        << "max diff = " << maxDiff << std::endl
        << "elapsed time = " << double(end - begin) / CLOCKS_PER_SEC << std::endl; 

    //auto cores = tt.Cores();

}