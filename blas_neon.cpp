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

// Optimized NEON implementation with multiple accumulators and loop unrolling
double ddot_neon_optimized(const std::vector<double> &x, const std::vector<double> &y) {
    size_t n = x.size();
    size_t i = 0;

    // Use 4 accumulators to hide FMA latency
    float64x2_t sum_vec0 = vdupq_n_f64(0.0);
    float64x2_t sum_vec1 = vdupq_n_f64(0.0);
    float64x2_t sum_vec2 = vdupq_n_f64(0.0);
    float64x2_t sum_vec3 = vdupq_n_f64(0.0);

    // Process 8 doubles at a time (4 vectors of 2 doubles each)
    for (; i + 7 < n; i += 8) {
        float64x2_t x_vec0 = vld1q_f64(&x[i]);
        float64x2_t y_vec0 = vld1q_f64(&y[i]);
        sum_vec0 = vfmaq_f64(sum_vec0, x_vec0, y_vec0);

        float64x2_t x_vec1 = vld1q_f64(&x[i + 2]);
        float64x2_t y_vec1 = vld1q_f64(&y[i + 2]);
        sum_vec1 = vfmaq_f64(sum_vec1, x_vec1, y_vec1);

        float64x2_t x_vec2 = vld1q_f64(&x[i + 4]);
        float64x2_t y_vec2 = vld1q_f64(&y[i + 4]);
        sum_vec2 = vfmaq_f64(sum_vec2, x_vec2, y_vec2);

        float64x2_t x_vec3 = vld1q_f64(&x[i + 6]);
        float64x2_t y_vec3 = vld1q_f64(&y[i + 6]);
        sum_vec3 = vfmaq_f64(sum_vec3, x_vec3, y_vec3);
    }

    // Combine the 4 accumulators
    sum_vec0 = vaddq_f64(sum_vec0, sum_vec1);
    sum_vec2 = vaddq_f64(sum_vec2, sum_vec3);
    sum_vec0 = vaddq_f64(sum_vec0, sum_vec2);

    // Process remaining elements in pairs
    for (; i + 1 < n; i += 2) {
        float64x2_t x_vec = vld1q_f64(&x[i]);
        float64x2_t y_vec = vld1q_f64(&y[i]);
        sum_vec0 = vfmaq_f64(sum_vec0, x_vec, y_vec);
    }

    // Horizontal add using vpaddq_f64 (more efficient than lane extraction)
    float64x2_t sum_pair = vpaddq_f64(sum_vec0, sum_vec0);
    double result = vgetq_lane_f64(sum_pair, 0);

    // Handle remaining element if size is odd
    for (; i < n; ++i) {
        result += x[i] * y[i];
    }

    return result;
}
