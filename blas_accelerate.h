#ifndef blas_accelerate_H
#define blas_accelerate_H

#include <vector>

double ddot_accelerate(const std::vector<double> &x, const std::vector<double> &y);

// Matrix multiplication: C = A * B
// A is m x k, B is k x n, C is m x n
void dgemm_accelerate(const std::vector<double> &A, const std::vector<double> &B,
                      std::vector<double> &C, size_t m, size_t n, size_t k);

#endif // blas_accelerate_H

