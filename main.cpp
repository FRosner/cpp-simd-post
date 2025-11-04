// main_benchmark.cpp

#include "blas_simple.cpp"
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

BENCHMARK(BM_DdotSimple)->RangeMultiplier(10)->Range(1<<10, 1<<20);

BENCHMARK_MAIN();
