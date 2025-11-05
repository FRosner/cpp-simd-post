#include "blas_openblas.h"

#include <cblas.h>

double ddot_openblas(const std::vector<double> &x, const std::vector<double> &y) {
    // Use OpenBLAS cblas_ddot for double-precision dot product
    int n = static_cast<int>(x.size());
    return cblas_ddot(n, x.data(), 1, y.data(), 1);
}

void dgemm_openblas(const std::vector<double> &A, const std::vector<double> &B,
                    std::vector<double> &C, size_t m, size_t n, size_t k) {
    // Use OpenBLAS for matrix multiplication
    // cblas_dgemm performs C = alpha*A*B + beta*C
    // We use alpha=1.0 and beta=0.0 to compute C = A*B
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0,           // alpha
                A.data(), k,   // A and leading dimension of A
                B.data(), n,   // B and leading dimension of B
                0.0,           // beta
                C.data(), n);  // C and leading dimension of C
}

