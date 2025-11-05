#ifndef BLAS_OPENBLAS_H
#define BLAS_OPENBLAS_H

#include <vector>

double ddot_openblas(const std::vector<double> &x, const std::vector<double> &y);

// Matrix multiplication: C = A * B
// A is m x k, B is k x n, C is m x n
void dgemm_openblas(const std::vector<double> &A, const std::vector<double> &B,
                    std::vector<double> &C, size_t m, size_t n, size_t k);

#endif // BLAS_OPENBLAS_H

