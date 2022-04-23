#include "matrix.hpp"
#include "calculation.hpp"



TMatrix::TMatrix(size_t n, long double elem) : TMatrix(n, n, elem) {}



TMatrix::TMatrix(size_t i, size_t j, long double elem) {
    rows_ = i;
    cols_ = j;
    data_ = new long double* [rows_];
    for (size_t c = 0;  c != rows_; c++)
        data_[c] = new long double [cols_];

    for (size_t i = 0; i != rows_; i++) {
        for (size_t j = 0; j != cols_; j++)
            data_[i][j] = elem;
    }
}

TMatrix::TMatrix(TMatrix const & other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = new long double* [rows_];
    for (size_t i = 0; i != rows_; i++)
        data_[i] = new long double [cols_];
    
    for (size_t i = 0; i != rows_; i++) {
        for (size_t j = 0; j != cols_; j++)
            data_[i][j] = other.data_[i][j];
    }
}


TMatrix::~TMatrix() {
    for (size_t i = 0; i != rows_; i++)
        delete [] data_[i];
    
    delete [] data_;
}


long double* TMatrix::operator [] (size_t i) {
    return data_[i];
}

long double const* TMatrix::operator [] (size_t i) const {
    return data_[i];
}


TMatrix TMatrix::operator+ (TMatrix const & other) const {
    assert(rows_ == other.rows_ and cols_ == other.cols_);
    TMatrix tmp = *this;
    for (size_t i = 0; i != rows_; i++) {
        for (size_t j = 0; j != cols_; j++)
            tmp.data_[i][j] += other.data_[i][j];
    }
    return tmp;
}


TMatrix TMatrix::operator- (TMatrix const & other) const{
    assert(rows_ == other.rows_ and cols_ == other.cols_);
    TMatrix tmp = *this;
    for (size_t i = 0; i != rows_; i++) {
        for (size_t j = 0; j != cols_; j++)
            tmp.data_[i][j] -= other.data_[i][j];
    }
    return tmp;
}

TMatrix TMatrix::operator* (TMatrix const & other) const{
    assert(cols_ == other.rows_);
    size_t l = rows_;
    size_t n = other.cols_;
    size_t m = cols_;
    TMatrix res(l, n);

    for (size_t i = 0; i != l; i++) {
        for (size_t j = 0; j != n; j++) {
            long double tmp = 0.;
            for (size_t k = 0; k != m; k++)
                tmp += data_[i][k] * other.data_[k][j];
            res.data_[i][j] = tmp;
        }
    }
    return res;
}

TMatrix TMatrix::operator *= (TMatrix const & other) {
    return (*this * other);
}


TMatrix TMatrix::operator* (int c) const {
    TMatrix res = *this;
    for (size_t i = 0; i != rows_; i++) {
        for (size_t j = 0; j != cols_; j++)
            res.data_[i][j] *= c;
    }
    return res;
}

TMatrix TMatrix::operator * (long double d) const {
    TMatrix res = *this;
    for (size_t i = 0; i != rows_; i++) {
        for (size_t j = 0; j != cols_; j++)
            res.data_[i][j] *= d;
    }
    return res;
}


TMatrix& TMatrix::operator= (TMatrix const& other) {
    this->~TMatrix();

    rows_ = other.rows_;
    cols_ = other.cols_;

    data_ = new long double* [rows_];
    for (size_t i = 0; i != rows_; i++)
        data_[i] = new long double [cols_];

    for (size_t i = 0; i != rows_; i++) {
        for (size_t j = 0; j != cols_; j++)
            data_[i][j] = other.data_[i][j];
    }

    return *this;




}

std::ostream& operator << (std::ostream& out, TMatrix const& m) {
    out << std::setprecision(5) << std::fixed;
    for (size_t i = 0; i != m.rows_; i++){
        for (size_t j = 0; j != m.cols_; j++) {
            out << m.data_[i][j] << ' ';
        }
        out << '\n';
    }
    return out;
}




std::istream& operator >> (std::istream& in, TMatrix & m) {
    for (size_t i = 0; i != m.rows_; i++) {
        for (size_t j = 0; j != m.cols_; j++)
            in >> m.data_[i][j];
    }
    return in;
}




bool TMatrix::operator== (TMatrix const& other) const {
    if (rows_ != other.rows_ or cols_ != other.cols_)
        return false;
    
    for (size_t i = 0; i != rows_; i++) {
        for (size_t j = 0; j != cols_; j++) {
            if (std::abs(data_[i][j] - other.data_[i][j]) > delta)
                return false;
        }
    }
    return true;
}

std::tuple<TMatrix, TMatrix, std::vector<std::pair<size_t, size_t>>> TMatrix::LUdecomposition() const {
    assert(rows_ == cols_);
    size_t n = rows_;
    TMatrix U = *this;
    TMatrix L(n);
    std::vector<std::pair<size_t, size_t>> P;
    
    
    
    for (size_t i = 0; i != n - 1; i++) {
        auto p = U.Change_With_Max(i);
        if (p.first != p.second)
            L.Swap_Rows(p.first, p.second);
        P.push_back(p);
        for (size_t j = i + 1; j != n; j++) {
            long double t = U.data_[j][i] / U.data_[i][i];
            for (size_t k = i; k != n; k++) {
                U.data_[j][k] -= t * U.data_[i][k];
            }
            L.data_[j][i] = t;
        }
    }

    for (size_t i = 0; i != n; i++)
        L.data_[i][i] = 1;

    return  std::make_tuple(L, U, P);
}

std::tuple<TMatrix, TMatrix> TMatrix::QRdecomposition() const {
    size_t n = Size();
    TMatrix A = *this;
    TMatrix E(n);
    for (size_t i = 0; i != n; i++)
        E[i][i] = 1.;

    TMatrix Q = E;
    TMatrix R(n);

    for (size_t k = 0; k != n - 1; k++) {
        
        TMatrix b(n - k, (size_t)1);
        for (size_t i = 0; i != n - k; i++)
            b[i][0] = A[k + i][k];

        TMatrix v(n, (size_t)1);
        for (size_t i = 0; i != n; i++) {
            if (i < k)
                v[i][0] = 0.;
            else if (i == k)
                v[i][0] = A[i][i] + Sign(A[i][i]) * b.Norm_2();
            else 
                v[i][0] = A[i][k];
        }

        TMatrix H = E - ((v * v.Transpose()) * (2. / TMatrix(v.Transpose() * v)[0][0]));
        A = H * A;
        Q = Q * H;
    }
    R = A;
    return std::make_tuple(Q, R);
    
}

long double TMatrix::Determinant() const {
    auto [L, U, P] = this->LUdecomposition();
    return LU_Determinant(U, P);
}


TMatrix TMatrix::Inverse() const {
    auto [L, U, P] = this->LUdecomposition();
    return LU_Inverse_Matrix(L, U, P);
}

TMatrix TMatrix::Transpose() const {
    size_t n = Get_Rows();
    size_t m = Get_Cols();
    TMatrix t(m ,n);
    for (size_t i = 0; i != rows_; i++) {
        for (size_t j = 0; j != cols_; j++) {
            t[j][i] = data_[i][j];
        }
    }
    return t;
}

size_t TMatrix::Size() const {
    assert(rows_ == cols_);
    return rows_;
}

size_t TMatrix::Get_Rows() const {
    return rows_;
}

size_t TMatrix::Get_Cols() const {
    return cols_;
}


long double  TMatrix::Norm() const {
    long double norm = -1.;
    
    for (size_t i = 0; i != rows_; i++) {
        long double sum = 0.;
        for (size_t j = 0; j != cols_; j++)
            sum += std::abs(data_[i][j]);
        norm = sum > norm ? sum : norm;
    }
    return norm;
}

long double TMatrix::Norm_2() const {
    long double sum = 0.;
    for (size_t i = 0; i != rows_; i++) {
        for (size_t j = 0; j != cols_; j++)
            sum += std::pow(data_[i][j], 2);
    }
    sum = std::pow(sum, 0.5);
    return sum;
}

long double TMatrix::GetSquaredColumnSum(size_t row, size_t col) const {
    long double sum = 0.;
    for (size_t j = row; j < rows_; j++) {
        sum += data_[j][col] * data_[j][col];
    }

    return std::sqrt(sum);

}

std::pair<size_t, size_t> TMatrix::Change_Without_Zero(size_t i) {
    size_t j = 0;
    for (size_t j = i + 1; j < Size(); j++) {
        if (data_[j][i] != 0.) {
            Swap_Rows(i ,j);
            break;
        }
    }
    return std::make_pair(i, j);    
}


std::pair<size_t, size_t> TMatrix::Change_With_Max(size_t i) {
    size_t pos_max = i;
    long double max = data_[i][i];
    for (size_t j = i + 1; j < Size(); j++)  {
        if (std::abs(data_[j][i]) > std::abs(max)) {
            max = data_[j][i];
            pos_max = j;
        }
    }
    Swap_Rows(i, pos_max);
    return std::make_pair(i, pos_max);
}



void TMatrix::Swap_Rows(size_t i, size_t j) {
    long double * tmp = new long double [cols_];
    for (size_t k = 0; k != cols_; k++)
        tmp[k] = data_[i][k];
    for (size_t k = 0; k != cols_; k++)
        data_[i][k] = data_[j][k];
    for (size_t k = 0; k != cols_; k++)
        data_[j][k] = tmp[k];
    delete [] tmp;
}

TMatrix TMatrix::leftSubmatrix(vector<int> const& I) const {
    int r = I.size(), m = this->Get_Cols();
    TMatrix res(r,m,0.);
    
    for (int i = 0; i < r; i++)
        for (int j = 0; j < m; j++)
            res[i][j] = this->data_[I[i]][j];

    return res;
}

TMatrix TMatrix::rightSubmatrix(vector<int> const& J) const {
    int n = this->Get_Rows(), r = J.size();
    TMatrix res(n,r,0.);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < r; j++)
            res[i][j] = this->data_[i][J[j]];

    return res;
}

TMatrix TMatrix::submatrix(vector<int> const &I, vector<int> const &J) const {
    int r = I.size();
    TMatrix res(r, r, 0.);

    for (int i = 0; i < r; i++)
        for (int j = 0; j < r; j++)
            res[i][j] = this->data_[I[i]][J[j]];

    return res;
}