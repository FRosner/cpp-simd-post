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
BENCHMARK(BM_DdotAccelerate)->RangeMultiplier(2)->Range(1<<10, 1<<22);
#endif
#ifdef USE_OPENBLAS
BENCHMARK(BM_DdotOpenBLAS)->RangeMultiplier(2)->Range(1<<10, 1<<22);
#endif

BENCHMARK_MAIN();
