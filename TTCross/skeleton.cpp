#include "skeleton.hpp"

pair<int,int> maxElem(TMatrix A, list<int> const& I, list<int> const& J) {
    double max = 0.;
    int iMax = max, jMax = 0;

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

pair<size_t,size_t> maxElem(const UnfoldingMatrix& A, const std::list<size_t>& I, const std::list<size_t>& J) {
    double max = 0.;
    size_t iMax = -1, jMax = -1;

    for (std::list<size_t>::const_iterator i = I.begin(); i != I.end(); i++) {
        for (std::list<size_t>::const_iterator j = J.begin(); j != J.end(); j++) {
            double tmp = abs(A(*i,*j));
            if (tmp > max) {
                max = tmp;
                iMax = *i;
                jMax = *j;
            }
        }
    }

    return std::make_pair(iMax,jMax);
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

        I.remove(iMax);
        J.remove(jMax);
        IMax.push_back(iMax);
        JMax.push_back(jMax);

        norm = eps * UV.Norm_2();
    } while(norm < fabs(a_ij) * sqrt((m - r) * (n - r)));

    return make_tuple(IMax, JMax);
}

std::pair<std::vector<size_t>,std::vector<size_t>> Skeleton(UnfoldingMatrix A, size_t maxR, double eps) {
    size_t n = A.N(), m = A.M();

    std::list<size_t> I(n), J(m);
    std::iota(I.begin(), I.end(), 0);
    std::iota(J.begin(), J.end(), 0);

    std::vector<size_t> IMax;
    std::vector<size_t> JMax;

    TMatrix UV;
    double norm, a_ij, criteria;

    size_t r = 0;
    while(r < maxR) {
        r++;
        
        std::pair<size_t,size_t> idxs = maxElem(A, I,J);
        if (idxs.first == size_t(-1) || idxs.second == size_t(-1)) break;

        TMatrix C(n,1,0.);
        for (size_t i = 0; i < n; i++) {
            C[i][0] = A(i,idxs.second);
        }

        TMatrix R(1,m,0.);
        for (size_t j = 0; j < m; j++) {
            R[0][j] = A(idxs.first,j);
        }

        a_ij = A(idxs.first,idxs.second);
        UV = C * TMatrix(1,1, 1/a_ij) * R;

        norm = eps * UV.Norm_2();
        criteria = fabs(a_ij) * sqrt((m - r) * (n - r));

        I.remove(idxs.first);
        J.remove(idxs.second);

        IMax.push_back(idxs.first);
        JMax.push_back(idxs.second);

        if (norm >= criteria) break;
    }

    std::sort(IMax.begin(),IMax.end());
    std::sort(JMax.begin(),JMax.end());
    
    return std::make_pair(IMax,JMax);
}