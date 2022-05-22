#ifndef DEV_CORE_CUH
#define DEV_CORE_CUH

class DevCore {
private:
    size_t a_, b_, c_;
    double* devData_{};

public:
    DevCore();
    DevCore(const double* src, size_t r_prev, size_t n_k, size_t r_k);
    DevCore(const DevCore& other);
    ~DevCore();

    size_t GetA() const {return a_;};
    size_t GetB() const {return b_;};
    size_t GetC() const {return c_;};

    double* Matrix(size_t i) const;

    DevCore& operator= (const DevCore& other);
};

#endif //DEV_CORE_CUH
