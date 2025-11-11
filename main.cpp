// main_benchmark.cpp

#include <benchmark/benchmark.h>
#include <vector>

// Conditional includes for BLAS libraries
#ifdef USE_ACCELERATE
#if !defined(__aarch64__) || !defined(__APPLE__)
#error "Accelerate implementation requires Apple Silicon (aarch64 on macOS)"
#endif
#include <Accelerate/Accelerate.h>
#define BLAS_BACKEND_NAME "Accelerate"
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#define BLAS_BACKEND_NAME "OpenBLAS"
#endif

// Unified BLAS wrapper functions
inline double blas_ddot(const std::vector<double> &x, const std::vector<double> &y) {
    int n = static_cast<int>(x.size());
    return cblas_ddot(n, x.data(), 1, y.data(), 1);
}

inline void blas_dgemm(const std::vector<double> &A, const std::vector<double> &B,
                       std::vector<double> &C, size_t m, size_t n, size_t k) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k,
                1.0,           // alpha
                A.data(), k,   // A and leading dimension of A
                B.data(), n,   // B and leading dimension of B
                0.0,           // beta
                C.data(), n);  // C and leading dimension of C
}

inline void blas_dgemv(const std::vector<double> &A, const std::vector<double> &x,
                       std::vector<double> &y, size_t m, size_t n) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                m, n,
                1.0,           // alpha
                A.data(), n,   // A and leading dimension of A
                x.data(), 1,   // x and stride
                0.0,           // beta
                y.data(), 1);  // y and stride
}

inline void blas_daxpy(const std::vector<double> &x, std::vector<double> &y, double alpha) {
    int n = static_cast<int>(x.size());
    cblas_daxpy(n, alpha, x.data(), 1, y.data(), 1);
}

// Macro to create benchmark function with dynamic naming
#define BENCHMARK_DDOT(backend_name) \
static void BM_Ddot_##backend_name(benchmark::State& state) { \
    size_t n = state.range(0); \
    std::vector<double> x(n, 1.0); \
    std::vector<double> y(n, 2.0); \
    for (auto _ : state) { \
        benchmark::DoNotOptimize(blas_ddot(x, y)); \
    } \
    state.SetItemsProcessed(state.iterations() * n); \
} \
BENCHMARK(BM_Ddot_##backend_name)->RangeMultiplier(2)->Range(1<<1, 1<<22);

#define BENCHMARK_DGEMM(backend_name) \
static void BM_Dgemm_##backend_name(benchmark::State& state) { \
    size_t n = state.range(0); \
    std::vector<double> A(n * n, 1.0); \
    std::vector<double> B(n * n, 2.0); \
    std::vector<double> C(n * n, 0.0); \
    for (auto _ : state) { \
        blas_dgemm(A, B, C, n, n, n); \
        benchmark::DoNotOptimize(C.data()); \
    } \
    state.SetItemsProcessed(state.iterations() * 2 * n * n * n); \
    state.SetBytesProcessed(state.iterations() * 3 * n * n * sizeof(double)); \
} \
BENCHMARK(BM_Dgemm_##backend_name)->RangeMultiplier(2)->Range(1<<1, 1<<13);

#define BENCHMARK_DGEMV(backend_name) \
static void BM_Dgemv_##backend_name(benchmark::State& state) { \
    size_t n = state.range(0); \
    std::vector<double> A(n * n, 1.0); \
    std::vector<double> x(n, 2.0); \
    std::vector<double> y(n, 0.0); \
    for (auto _ : state) { \
        blas_dgemv(A, x, y, n, n); \
        benchmark::DoNotOptimize(y.data()); \
    } \
    state.SetItemsProcessed(state.iterations() * 2 * n * n); \
    state.SetBytesProcessed(state.iterations() * (n * n + 2 * n) * sizeof(double)); \
} \
BENCHMARK(BM_Dgemv_##backend_name)->RangeMultiplier(2)->Range(1<<1, 1<<13);

#define BENCHMARK_DAXPY(backend_name) \
static void BM_Daxpy_##backend_name(benchmark::State& state) { \
    size_t n = state.range(0); \
    std::vector<double> x(n, 1.0); \
    std::vector<double> y(n, 2.0); \
    double alpha = 3.0; \
    for (auto _ : state) { \
        blas_daxpy(x, y, alpha); \
        benchmark::DoNotOptimize(y.data()); \
    } \
    state.SetItemsProcessed(state.iterations() * n); \
    state.SetBytesProcessed(state.iterations() * 3 * n * sizeof(double)); \
} \
BENCHMARK(BM_Daxpy_##backend_name)->RangeMultiplier(2)->Range(1<<1, 1<<22);

// Register benchmarks based on the selected backend
#ifdef USE_ACCELERATE
BENCHMARK_DDOT(Accelerate)
BENCHMARK_DGEMM(Accelerate)
BENCHMARK_DGEMV(Accelerate)
BENCHMARK_DAXPY(Accelerate)
#endif

#ifdef USE_OPENBLAS
BENCHMARK_DDOT(OpenBLAS)
BENCHMARK_DGEMM(OpenBLAS)
BENCHMARK_DGEMV(OpenBLAS)
BENCHMARK_DAXPY(OpenBLAS)
#endif

BENCHMARK_MAIN();
