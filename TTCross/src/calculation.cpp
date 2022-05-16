#include <vector>
#include <algorithm>
#include <limits>

#include "../include/calculation.hpp"

//Task1
TMatrix LU_Solving_System(TMatrix const& L, TMatrix const& U, TMatrix b, std::vector<std::pair<size_t, size_t>> const& p) {
    for (size_t i = 0; i != p.size(); i++)
        b.Swap_Rows(p[i].first, p[i].second);
    //Ly = b
    //Forward Substitution

    size_t n = L.Size();
    TMatrix y(n, size_t(1));
    for (size_t i = 0; i != n; i++) {
        long double t = 0.;
        for (size_t j = 0; j != i; j++)
           t += y[j][0] * L[i][j];
        y[i][0] = b[i][0] - t;
    }
    
    //Ux = y;
    //Back Substitution

    TMatrix x(n, (size_t)1);
    for (int i = n - 1; i >= 0; i--) {
        long double t = 0.;
        for (int j = i + 1; j < (int)n; j++)
            t += U[i][j] * x[j][0];
        x[i][0] = (y[i][0] - t) / U[i][i];
    }

    return x;

}


long double LU_Determinant(TMatrix const& U, std::vector<std::pair<size_t, size_t>> const & P) {
    size_t p = 0;
    for (auto a : P)
        p = a.first != a.second ? p + 1 : p;

    long double det = 1.;
    for (size_t i = 0; i != U.Size(); i++)
        det *= U[i][i];
    return std::pow(-1, p) * det; 
}



TMatrix LU_Inverse_Matrix(TMatrix const& L, TMatrix const& U, std::vector<std::pair<size_t, size_t>> const& p) {
    size_t n = L.Size();
    TMatrix Inverse(n);
    for (size_t i = 0; i != n; i++) {
        TMatrix b(n, (size_t)1);
        b[i][0] = 1;
        TMatrix tmp = LU_Solving_System(L, U, b, p);
        for (size_t j = 0; j != n; j++)
            Inverse[j][i] = tmp[j][0];
    }
    return Inverse;
}

