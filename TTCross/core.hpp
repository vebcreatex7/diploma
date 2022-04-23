#include "matrix.hpp"

class Core {
private:
    size_t n_, m_, k_;
    double*** data;

public:
    Core(TMatrix A, size_t r_prev, size_t n_k, size_t r_k);
};