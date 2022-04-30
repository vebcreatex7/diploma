#include "calculation.hpp"
#include "matrix.hpp"
#include "skeleton.hpp"
#include "core.hpp"
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

int main() {
    size_t n, m;
    cin >> n >> m;

    TMatrix A = SparseMatrix(n,m, 0.5);

    size_t a, b, c;
    cin >> a >> b >> c;

    if (a * b * c != n * m) {
        cout << "a * b * c != n * m\n";
        return 0;
    }

    Core C(A, a,b,c);

    for (size_t i = 0; i < a; i++) {
        for (size_t j = 0; j < b; j++) {
            for (size_t k = 0; k < c; k++) {
                cout << C(i,j,k) << ' ';
            }
            cout << endl;
        }
        cout << endl;
    }
}