#include "blas_amx.h"

#if !defined(__aarch64__) || !defined(__APPLE__)
#error "AMX implementation requires Apple Silicon (aarch64 on macOS)"
#endif

#include <Accelerate/Accelerate.h>

double ddot_amx(const std::vector<double> &x, const std::vector<double> &y) {
    // Use Apple's Accelerate framework which leverages AMX on supported hardware
    // cblas_ddot performs double-precision dot product
    int n = static_cast<int>(x.size());
    return cblas_ddot(n, x.data(), 1, y.data(), 1);
}

