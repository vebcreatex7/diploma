#include "matrix.hpp"
#include <tuple>
#include <iomanip>

int Diagonals_Without_Zeros(TMatrix& A, TMatrix& b);

//Task1
TMatrix LU_Solving_System(TMatrix const& L, TMatrix const& U, TMatrix b, std::vector<std::pair<size_t, size_t>> const& p);
long double LU_Determinant(TMatrix const& U, std::vector<std::pair<size_t, size_t>> const&);
TMatrix LU_Inverse_Matrix(TMatrix const& L, TMatrix const& U, std::vector<std::pair<size_t, size_t>> const& p);


//Task2
TMatrix Tridiagonal_Algorithm(TMatrix const& A, TMatrix const& D);

//Task3

std::tuple<TMatrix, int, int, long double> Iterative_Jacobi_Method(TMatrix const& A, TMatrix const& b, long double eps, std::ostream& log);
std::tuple<TMatrix, int, int, long double> Seidel_Method(TMatrix const& A, TMatrix const& b, long double const eps, std::ostream& log);

//Task4

std::tuple<std::vector<long double>, std::vector<TMatrix>, size_t> Rotation_Method(TMatrix const& M, long double const eps, std::ostream& log);

//Task 5
int Sign(long double d);
std::vector<std::complex<long double>> Solve_Quadratic_Equation(TMatrix const& A, size_t col);
std::tuple<std::vector<std::complex<long double>>, int> Eigenvalues_Using_QR(TMatrix const& A, long double eps, std::ostream& log);