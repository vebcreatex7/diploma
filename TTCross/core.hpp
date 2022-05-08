#ifndef CORE_H
#define CORE_H

#include "matrix.hpp"

class Core {
private:
    size_t a_, b_, c_;  
    double* data_;

public:
    Core();
    Core(const TMatrix& A, size_t r_prev, size_t n_k, size_t r_k);
    Core(const Core& other);
    ~Core();
    std::tuple<size_t,size_t,size_t> Sizes() const;
    double operator()(size_t i, size_t j, size_t k) const;
    Core& operator= (const Core& other);
    TMatrix operator()(size_t j) const;

    void print() const;
    friend std::ostream& operator<< (std::ostream& os, const Core& c);
};

#endif