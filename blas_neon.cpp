#include "blas_neon.h"
#include <arm_neon.h>

double ddot_neon(const std::vector<double> &x, const std::vector<double> &y) {
    size_t n = x.size();
    size_t i = 0;
    float64x2_t sum_vec = vdupq_n_f64(0.0);

    // Process two doubles at a time
    for (; i + 1 < n; i += 2) {
        float64x2_t x_vec = vld1q_f64(&x[i]);
        float64x2_t y_vec = vld1q_f64(&y[i]);
        sum_vec = vfmaq_f64(sum_vec, x_vec, y_vec); // sum_vec += x_vec * y_vec
    }

    // Horizontal add the 2 elements in sum_vec
    double result = vgetq_lane_f64(sum_vec, 0) + vgetq_lane_f64(sum_vec, 1);

    // Handle remaining element if size is odd
    for (; i < n; ++i) {
        result += x[i] * y[i];
    }

    return result;
}
