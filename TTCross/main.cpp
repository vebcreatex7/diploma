#include "calculation.hpp"
#include "matrix.hpp"
#include "skeleton.hpp"
#include <random>

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

int main() {
    srand (time(NULL));
    int n, m;
    double density;
    std::cin >> n >> m >> density;
    TMatrix A = SparseMatrix(n,m, density);
    cout << A << endl;
    auto [I,J] = Skeleton(A, 0.1);

    TMatrix U = RestoreCols(A, J);
    TMatrix V = RestoreRows(A, I);
    TMatrix A_hat = maxvol(A, I, J);

    //std::cout << (A - (U * A_hat.Inverse() * V)).Norm_2();
    //std::cout << A << "\n" << U * A_hat.Inverse() * V << "\n";
    cout << I.size();
    
}