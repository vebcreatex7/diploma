#include "matrix.hpp"

class Core {
private:
    size_t a_, b_, c_;  
    double* data_;

public:
    Core(TMatrix A, size_t r_prev, size_t n_k, size_t r_k);
    ~Core() {free(data_);}
    double operator()(size_t i, size_t j, size_t k) const;
};