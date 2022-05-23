#ifndef CORE_H
#define CORE_H

#include <vector>

#include "matrix.cuh"
#include "dev_matrix.cuh"

class Core {
private:
    size_t a_, b_, c_;
    double* data_{};

public:
    Core();
    Core(const TMatrix& A, size_t r_prev, size_t n_k, size_t r_k);
    Core(const Core& other);
    ~Core();

    size_t GetA() const {return a_;};
    size_t GetB() const {return b_;};
    size_t GetC() const {return c_;};


    double operator()(size_t i, size_t j, size_t k) const;
    Core& operator= (const Core& other);
    TMatrix operator()(size_t j) const;
};

#endif