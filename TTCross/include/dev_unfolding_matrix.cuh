#ifndef DEV_UNFOLDING_MATRIX_CUH
#define DEV_UNFOLDING_MATRIX_CUH

#include <vector>

#include "dev_implicit_tensor.cuh"
#include "dev_matrix.cuh"

class DevUnfoldingMatrix {
private:
    size_t rows_, cols_;
    DevImplicitTensor* t_;

public:
    DevUnfoldingMatrix();
    DevUnfoldingMatrix(DevImplicitTensor& t, size_t rows, size_t cols);

    size_t Rows() const {return rows_;}
    size_t Cols() const {return cols_;}

    double* ExplicitRows(const std::vector<size_t>& I) const;
    double* ExplicitCols(const std::vector<size_t>& J) const;
    double* ExplicitMaxvol(const std::vector<size_t>& I, const std::vector<size_t>& J) const;

    void print() const;

    double operator() (size_t i, size_t j) const;
    DevUnfoldingMatrix& operator= (const DevUnfoldingMatrix& other);

    friend std::ostream& operator<< (std::ostream& os, const DevUnfoldingMatrix& m);

};

#endif //DEV_UNFOLDING_MATRIX_CUH
