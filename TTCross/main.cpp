
#include "tensorTrain.hpp"
#include <random>
#include <vector>

using namespace std;

double density = 0.05;


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

TMatrix RestoreRows(TMatrix& a, vector<int>& I) {
    int m = a.Get_Cols();
    int r = I.size();

    TMatrix res(r,m,0.);

    for (int i = 0; i < r; i++) {
        for (int j = 0; j < m; j++)
            res[i][j] = a[I[i]][j];
    }

    return res;
}

TMatrix RestoreCols(TMatrix& a, vector<int>& J) {
    int n = a.Get_Rows();
    int r = J.size();

    TMatrix res(n,r,0.);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < r; j++)
            res[i][j] = a[i][J[j]];
    }

    return res;
}

TMatrix maxvol(TMatrix& a, vector<int>& I, vector<int>& J){
    int r = I.size();

    TMatrix res(r,r,0.);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < r; j++)
            res[i][j] = a[I[i]][J[j]];
    }

    return res;
}

double f(const std::vector<size_t> &idxs) {
    return  sin(std::accumulate(idxs.begin(), idxs.end(), (size_t)0));
}

int main() {
    srand(time(NULL));
    size_t d = 2;
    size_t n = 50, m = 50;
    double eps = 0.1;
    TMatrix A = SparseMatrix(n,m, 0.01);

    std::vector<size_t> s{n,m};

    auto fp = std::bind(&TMatrix::f, A, std::placeholders::_1);
    ImplicitTensor t = ImplicitTensor(d, s, fp);

    TensorTrain tt;

    tt.TTCross(t, eps);
    std::vector<Core> cores = tt.Cores();

    std::vector<size_t> ttRanks = tt.TTRanks();
    for (auto a : ttRanks) {
        std::cout << a << ' ';
    }
    std::cout << std::endl;

    auto [I, J] = Skeleton(A, eps);
    std::cout << I.size() << std::endl;

}