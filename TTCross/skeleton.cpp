#include "skeleton.hpp"

pair<int,int> maxElem(TMatrix A, list<int> const& I, list<int> const& J) {
    double max = 0.;
    int iMax = 0, jMax = 0;

    for (list<int>::const_iterator i = I.begin(); i != I.end(); i++) {
        for (list<int>::const_iterator j = J.begin(); j != J.end(); j++) {
            if (abs(A[*i][*j]) > max) {
                max = abs(A[*i][*j]);
                iMax = *i;
                jMax = *j;
            }
        }
    }

    return make_pair(iMax, jMax);
}

tuple<vector<int>, vector<int>> Skeleton(TMatrix A, double eps) {
    int n = A.Get_Rows(), m = A.Get_Cols();

    list<int> I(n), J(m);
    iota(I.begin(), I.end(), 0);
    iota(J.begin(), J.end(), 0);

    vector<int> IMax, JMax;

    TMatrix UV;
    int iMax, jMax;
    double norm, a_ij;

    int r = 0;
    do {
        r++;
        pair<int, int> idxs = maxElem(A, I, J);
        iMax = idxs.first, jMax = idxs.second;

        TMatrix C(n, 1, 0.);
        for (int i = 0; i < n; i++)
            C[i][0] = A[i][jMax];

        TMatrix R(1, m, 0.);
        for (int j = 0; j < m; j++)
            R[0][j] = A[iMax][j];

        a_ij = A[iMax][jMax];
        if (a_ij == 0.) break;
        UV = C * TMatrix(1,1, 1/a_ij) * R;

        //A = A - UV;

        I.remove(iMax);
        J.remove(jMax);
        IMax.push_back(iMax);
        JMax.push_back(jMax);

        norm = eps * UV.Norm_2();
    } while(norm < fabs(a_ij) * sqrt((m - r) * (n - r)));

    return make_tuple(IMax, JMax);
}

tuple<vector<size_t>, vector<size_t>> Skeleton(UnfoldingMatrix A, double eps) {
    size_t s = min(A.N(), A.M());

    vector<size_t> I(s/2), J(s/2);

    for (size_t i = 0; i < s/2; i++) {
        I[i] = 2*i;
        J[i] = 2*i + 1;
    }

    return make_tuple(I,J);
}