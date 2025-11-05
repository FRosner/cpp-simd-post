#ifndef BLAS_NEON_H
#define BLAS_NEON_H

#include <vector>

double ddot_neon(const std::vector<double> &x, const std::vector<double> &y);
double ddot_neon_optimized(const std::vector<double> &x, const std::vector<double> &y);

#endif // BLAS_NEON_H

