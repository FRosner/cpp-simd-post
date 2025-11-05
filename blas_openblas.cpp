#include "blas_openblas.h"

#include <cblas.h>

double ddot_openblas(const std::vector<double> &x, const std::vector<double> &y) {
    // Use OpenBLAS cblas_ddot for double-precision dot product
    int n = static_cast<int>(x.size());
    return cblas_ddot(n, x.data(), 1, y.data(), 1);
}

