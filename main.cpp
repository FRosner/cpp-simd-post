// main_benchmark.cpp

#ifdef USE_ACCELERATE
#include "blas_accelerate.h"
#endif
#ifdef USE_OPENBLAS
#include "blas_openblas.h"
#endif
#include <benchmark/benchmark.h>
#include <vector>

#ifdef USE_ACCELERATE
static void BM_DdotAccelerate(benchmark::State& state) {
    size_t n = state.range(0);
    std::vector<double> x(n, 1.0);
    std::vector<double> y(n, 2.0);
    for (auto _ : state) {
        // Measure AMX-accelerated dot product performance (Apple Accelerate)
        benchmark::DoNotOptimize(ddot_accelerate(x, y));
    }
    state.SetItemsProcessed(state.iterations() * n);
}
#endif

#ifdef USE_OPENBLAS
static void BM_DdotOpenBLAS(benchmark::State& state) {
    size_t n = state.range(0);
    std::vector<double> x(n, 1.0);
    std::vector<double> y(n, 2.0);
    for (auto _ : state) {
        // Measure OpenBLAS dot product performance
        benchmark::DoNotOptimize(ddot_openblas(x, y));
    }
    state.SetItemsProcessed(state.iterations() * n);
}
#endif

#ifdef USE_ACCELERATE
static void BM_DgemmAccelerate(benchmark::State& state) {
    size_t n = state.range(0);  // Matrix dimension (n x n matrices)
    std::vector<double> A(n * n, 1.0);
    std::vector<double> B(n * n, 2.0);
    std::vector<double> C(n * n, 0.0);
    for (auto _ : state) {
        // Measure AMX-accelerated matrix multiplication performance (Apple Accelerate)
        dgemm_accelerate(A, B, C, n, n, n);
        benchmark::DoNotOptimize(C.data());
    }
    // For matrix multiplication, we perform 2*n^3 FLOPs (n^3 multiplications + n^3 additions)
    state.SetItemsProcessed(state.iterations() * 2 * n * n * n);
    state.SetBytesProcessed(state.iterations() * 3 * n * n * sizeof(double));
}
#endif

#ifdef USE_OPENBLAS
static void BM_DgemmOpenBLAS(benchmark::State& state) {
    size_t n = state.range(0);  // Matrix dimension (n x n matrices)
    std::vector<double> A(n * n, 1.0);
    std::vector<double> B(n * n, 2.0);
    std::vector<double> C(n * n, 0.0);
    for (auto _ : state) {
        // Measure OpenBLAS matrix multiplication performance
        dgemm_openblas(A, B, C, n, n, n);
        benchmark::DoNotOptimize(C.data());
    }
    // For matrix multiplication, we perform 2*n^3 FLOPs (n^3 multiplications + n^3 additions)
    state.SetItemsProcessed(state.iterations() * 2 * n * n * n);
    state.SetBytesProcessed(state.iterations() * 3 * n * n * sizeof(double));
}
#endif

#ifdef USE_ACCELERATE
BENCHMARK(BM_DdotAccelerate)->RangeMultiplier(2)->Range(1<<10, 1<<22);
#endif
#ifdef USE_OPENBLAS
BENCHMARK(BM_DdotOpenBLAS)->RangeMultiplier(2)->Range(1<<10, 1<<22);
#endif

// Matrix multiplication benchmarks with smaller range (n x n matrices)
// Range from 64x64 to 2048x2048 matrices
#ifdef USE_ACCELERATE
BENCHMARK(BM_DgemmAccelerate)->RangeMultiplier(2)->Range(1<<6, 1<<11);
#endif
#ifdef USE_OPENBLAS
BENCHMARK(BM_DgemmOpenBLAS)->RangeMultiplier(2)->Range(1<<6, 1<<11);
#endif

BENCHMARK_MAIN();
