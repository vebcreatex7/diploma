
#include "tensorTrain.hpp"
#include <random>
#include <vector>

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
    return sin(std::accumulate(idxs.begin(), idxs.end(), (size_t)0));
}

int main() {
    size_t d = 5;
    std::vector<size_t> s{4,4,4,4,4};
    ImplicitTensor t = ImplicitTensor(d, s, f);

    TensorTrain tt;

    tt.TTCross(t, 0.1);
    std::vector<Core> cores = tt.Cores();

    for (auto& core : cores) {
        std::cout << core << std::endl;
        std::cout << "======================\n\n";
    }

}