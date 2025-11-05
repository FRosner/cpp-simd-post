// main_benchmark.cpp

#include "blas_simple.h"
#include "blas_neon.h"
#include "blas_amx.h"
#include <benchmark/benchmark.h>
#include <vector>

static void BM_DdotSimple(benchmark::State& state) {
    size_t n = state.range(0);
    std::vector<double> x(n, 1.0);
    std::vector<double> y(n, 2.0);
    for (auto _ : state) {
        // Measure dot product performance
        benchmark::DoNotOptimize(ddot_simple(x, y));
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_DdotNeon(benchmark::State& state) {
    size_t n = state.range(0);
    std::vector<double> x(n, 1.0);
    std::vector<double> y(n, 2.0);
    for (auto _ : state) {
        // Measure SIMD-accelerated dot product performance
        benchmark::DoNotOptimize(ddot_neon_optimized(x, y));
    }
    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_DdotAmx(benchmark::State& state) {
    size_t n = state.range(0);
    std::vector<double> x(n, 1.0);
    std::vector<double> y(n, 2.0);
    for (auto _ : state) {
        // Measure AMX-accelerated dot product performance
        benchmark::DoNotOptimize(ddot_amx(x, y));
    }
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_DdotSimple)->RangeMultiplier(8)->Range(1<<10, 1<<20);
BENCHMARK(BM_DdotNeon)->RangeMultiplier(8)->Range(1<<10, 1<<20);
BENCHMARK(BM_DdotAmx)->RangeMultiplier(8)->Range(1<<10, 1<<20);

BENCHMARK_MAIN();
