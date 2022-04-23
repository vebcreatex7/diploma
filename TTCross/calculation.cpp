#include "calculation.hpp"
#include <vector>
#include <algorithm>
#include <limits>

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



//Task2
TMatrix Tridiagonal_Algorithm(TMatrix const& A, TMatrix const& D) {
    size_t n = A.Get_Rows();
    std::vector<long double> P(n);
    std::vector<long double> Q(n);

    //Forward Substitution

    long double a = 0.;
    long double b = A[0][1];
    long double c = A[0][2];
    long double d = D[0][0];
    P[0] = -c / b;
    Q[0] = d / b;
    for (size_t i = 1; i != n - 1; i++) {
        a = A[i][0];
        b = A[i][1];
        c = A[i][2];
        d = D[i][0];
        P[i] = -c / (b + a * P[i - 1]);
        Q[i] = (d - a * Q[i - 1]) / (b + a * P[i - 1]);
    }
    P[n - 1] = 0;
    a = A[n - 1][0];
    b = A[n - 1][1];
    d = D[n - 1][0];
    Q[n - 1] = (d - a * Q[n - 2]) / (b + a * P[n - 2]);

    

    //Back Substitution

    TMatrix x(n, (size_t)1);
    x[n - 1][0] = Q[n - 1];
    for (int i = n - 2; i >= 0; i--)
        x[i][0] = P[i] * x[i + 1][0] + Q[i];

    return x;

}



//Task3

std::tuple<TMatrix, int, int, long double> Iterative_Jacobi_Method(TMatrix const& A, TMatrix const& b, long double eps, std::ostream& log) {
    size_t n = A.Size();
    TMatrix alpha = A;
    TMatrix beta = b;

    auto [L, U, P] = A.LUdecomposition();
    TMatrix Exact_Solution = LU_Solving_System(L, U, b, P);

    for (size_t i = 0; i != n; i++) {
        if(alpha[i][i] == 0) {
            auto t = alpha.Change_Without_Zero(i);
            beta.Swap_Rows(t.first, t.second);
        }
        long double tmp = alpha[i][i];
        beta[i][0] /= tmp;
        for (size_t j = 0; j != n; j++) {
            
            if (j == i)
                alpha[i][i] = 0;
            else
                alpha[i][j] /= -tmp;

        }
    }

    //Norm
    long double norm = alpha.Norm();

    //A priori estimation of the number of iterations
    int k = (int)ceil(((log10(eps) - log10(beta.Norm()) + log10(1 - norm)) / log10(norm)) - 1);

    //Iterations
    TMatrix x_prev = beta;
    TMatrix x;
    int count = 0;
    log << "x_" << count << '\n' <<  x_prev << '\n';
    while (true) {
        count++;
        x = beta + alpha * x_prev;


        long double eps_k;
        if (norm < 1.) {
            eps_k = (norm / (1 - norm)) * TMatrix(x - x_prev).Norm();
        } else {
            eps_k = TMatrix(x - x_prev).Norm();
        }

        log << "x_" << count << '\n' <<  x_prev 
        << "eps_" << count << " = " << eps_k << "\n\n";


        if (eps_k <= eps)
            break;
        
        x_prev = x;
    }
    
    
    return std::make_tuple(x, count, k, norm);
}

std::tuple<TMatrix, int, int, long double> Seidel_Method(TMatrix const& A, TMatrix const& b, long double const eps, std::ostream& log) {
    size_t n = A.Size();
    TMatrix alpha = A;
    TMatrix beta = b;
    for (size_t i = 0; i != n; i++) {
        if(alpha[i][i] == 0) {
            auto t = alpha.Change_Without_Zero(i);
            beta.Swap_Rows(t.first, t.second);
        }
        long double tmp = alpha[i][i];
        beta[i][0] /= tmp;
        for (size_t j = 0; j != n; j++) {
            
            if (j == i)
                alpha[i][i] = 0;
            else
                alpha[i][j] /= -tmp;

        }
    }

    TMatrix B(n), C(n);
    for (size_t i = 0; i != n ; i++) {
        for (size_t j = 0; j != n; j++) {
            if (i > j)
                B[i][j] = alpha[i][j];
            else {
                C[i][j] = alpha[i][j];
            }
        }
    }
    TMatrix E(n);
    for (size_t i = 0; i != n; i++)
        E[i][i] = 1;

    alpha = (E - B).Inverse() * C;
    beta = (E - B).Inverse() * beta;

    
    //Norm
    long double norm = alpha.Norm();

    //A priori estimation of the number of iterations
    int k = (int)ceil(((log10(eps) - log10(beta.Norm()) + log10(1 - norm)) / log10(norm)) - 1);



    //Iterations
    TMatrix x_prev = beta;
    TMatrix x;
    int count = 0;
    log << "x_" << count << '\n' <<  x_prev << '\n';
    while (true) {
        count++;
        x = beta + alpha * x_prev;

        long double eps_k;
        if (norm < 1.) {
            eps_k = (C.Norm() / (1 - norm)) * TMatrix(x - x_prev).Norm();
        } else {
            eps_k = TMatrix(x - x_prev).Norm();
        }

        log << "x_" << count << '\n' <<  x_prev 
        << "eps_" << count << " = " << eps_k << "\n\n";



        if (eps_k <= eps)
            break;
        x_prev = x;
    }

    return std::make_tuple(x, count, k, norm);

}




//Task4

long double t(TMatrix  const& A) {
    size_t n = A.Size();
    long double sum = .0;
    for (size_t i = 0; i != n; i++) {
        for (size_t j = 0; j != i; j++) {
            sum += std::pow(A[i][j], 2);
        }
    }
    return std::sqrt(sum);
}

std::tuple<std::vector<long double>, std::vector<TMatrix>, size_t> Rotation_Method(TMatrix const& M, long double const eps, std::ostream& log) {
    size_t n = M.Size();
    TMatrix A = M;
    TMatrix E(n);
    for (size_t i = 0; i != n; i++)
        E[i][i] = 1.;
    TMatrix U_Composition = E;

    //Iterations
    size_t count = 0;
    while (true) {
        //find the maximum non-diagonal
        long double max = -1e-10;
        size_t i_max = -1;
        size_t j_max = -1;
        for (size_t i = 0; i != n; i++) {
            for (size_t j = 0; j != i; j ++) {
                if (std::abs(A[i][j]) > std::abs(max)) {
                    max = A[i][j];
                    i_max = i;
                    j_max = j;
                }
            }
        }

        //Rotation angle
        long double phi = .0;
        if (std::abs(A[i_max][i_max] - A[j_max][j_max]) < delta)
            phi = M_PI_4;
        else
            phi = atan(2 * A[i_max][j_max] / (A[j_max][j_max] - A[i_max][i_max])) / 2;
        

        //Building matrix U
        TMatrix U = E;
        U[i_max][i_max] = cos(phi);
        U[j_max][j_max] = cos(phi);
        U[i_max][j_max] = sin(phi);
        U[j_max][i_max] = -sin(phi);

        U_Composition = U_Composition * U;
        

        //Iterate
        A = U.Transpose() * A * U;
        long double accurate = t(A);
        log << "max|a| = " << max << '\n'
            << "phi = " << phi << '\n'
            << "U_" << count << ":\n" << U
            << "A_" << count + 1 << ":\n" << A
            << "t = " << accurate << "\n\n";
        
        count++;

        if (accurate < eps)
            break;  
    }



    //Restor Eigenvalues and eigenvectors

    std::vector<long double> Eigenvalues(n);
    for (size_t i = 0; i != n; i++)
        Eigenvalues[i] = A[i][i];

    std::vector<TMatrix> Eigenvectors(n);
    for (size_t j = 0; j != n; j++) {
        TMatrix tmp(n, size_t(1));
        for (size_t i = 0; i != n; i++)
            tmp[i][0] = U_Composition[i][j];
        Eigenvectors[j] = tmp;
    }

    return std::make_tuple(Eigenvalues, Eigenvectors, count);

    
    
}


//Task 5

int Sign(long double d) {
    return d < 0 ? -1 : 1;
}


std::vector<std::complex<long double>> Solve_Quadratic_Equation(TMatrix const& A, size_t col) {
    std::vector<std::complex<long double>> v;
    long double b = -A[col][col] - A[col + 1][col + 1];
    long double c = A[col][col] * A[col + 1][col + 1] - (A[col][col + 1] * A[col + 1][col]);
    long double d = std::pow(b, 2.) - 4 * c;
    if (d > 0) {
        v.push_back(std::complex<long double>{(-b - std::sqrt(d)) / 2., 0});
        v.push_back(std::complex<long double>{(-b + std::sqrt(d)) / 2., 0});
    } else if (d == 0) {
        v.push_back(std::complex<long double>{-b / 2., 0});
        v.push_back(std::complex<long double>{-b / 2., 0});
    } else {
         v.push_back(std::complex<long double>{-b / 2., -sqrt(-d) / 2.});
         v.push_back(std::complex<long double>{-b / 2., sqrt(-d) / 2.});
    }
    return v;
}

std::tuple<std::vector<std::complex<long double>>, int> Eigenvalues_Using_QR(TMatrix const& A, long double eps, std::ostream& log) {
    std::vector<std::array<std::complex<long double>, 2>> Eigens(A.Get_Rows());
    TMatrix A_k = A;
    bool stop = true;
    size_t count = 0;
    while (stop) {
        count++;
        auto [Q, R] = A_k.QRdecomposition();
        A_k = R * Q;
        log << "Q_" << count - 1 << ":\n" << Q << "\n"
            << "R_" << count - 1 << ":\n" << R << "\n"
            << "A_" << count << ":\n" << A_k << "\n";
        
        stop = false;

        for (size_t i = 0 ;i < A_k.Size(); i++) {
            long double check = A_k.GetSquaredColumnSum(i + 1, i);

            if (check > eps) {
                auto Roots = Solve_Quadratic_Equation(A_k, i);
                if (std::abs(std::abs(Eigens[i][0]) - std::abs(Roots[0])) > eps)
                    stop = true;
                Eigens[i][0] = Roots[0];
                Eigens[i][1] = Roots[1];
                i++;
            } else {
                Eigens[i][0] = A_k[i][i];
                Eigens[i][1] = {};
            }

        }
    }

    std::vector<std::complex<long double>> result;
    for (size_t i = 0; i < Eigens.size(); i++) {
        if (std::abs(Eigens[i][1]) < eps) {
            result.push_back(Eigens[i][0]);
        } else {
            result.push_back(Eigens[i][0]);
            result.push_back(Eigens[i][1]);
            i++;
        }
    }
    return std::make_tuple(result, count);
}