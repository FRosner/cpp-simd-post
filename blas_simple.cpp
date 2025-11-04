#include <vector>

// Simple BLAS operation: vector dot product (DDOT)
double ddot_simple(const std::vector<double> &x, const std::vector<double> &y) {
    double result = 0.0;
    size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        result += x[i] * y[i];
    }
    return result;
}
