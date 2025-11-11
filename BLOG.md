---
title: Using Apple's Accelerate Framework for BLAS Routines
published: false
description: none 
tags:  none
# cover_image: https://direct_url_to_image.jpg
# Use a ratio of 100:42 for best results.
# published_at: 2025-08-11 07:06 +0000
---

## Motivation

Many real world applications such as machine learning, scientific computing, data compression, computer graphics and video processing require linear algebra operations. Tensors (mostly vectors, which are 1-dimensional tensors, and matrices, which are 2-dimensional tensors) are the primary data structures used to represent data in these applications.

Writing software is hard. Writing correct, performant, secure, reliable, etc., software is even harder. This is why most linear algebra operations are expressed in terms of Basic Linear Algebra Subprograms (BLAS). Common BLAS routines are vector addition, scalar multiplication, dot products, linear combinations, matrix-vector multiplication, and matrix-matrix multiplication.

Similarly to the saying "Don't roll your own cryptography", we should also heed the advice "Don't roll your own BLAS". BLAS libraries are often tuned for the specific SIMD (Single Instruction, Multiple Data) instructions of the underlying hardware. This can lead to performance improvements in orders of magnitude.

In this post we want to explore how to utilize BLAS libraries on Apple Silicon using C++, and compare the performance of OpenBLAS and Apple's Accelerate framework.

## A Brief History of SIMD

SIMD is a type of parallel computing first conceptually classified in Flynn's taxonomy[^flynn-1966] in the 1960s. In the 1990s, as desktop processors became more powerful, SIMD instructions were introduced to improve multimedia and gaming performance. Intel's MMX, released in 1996, was the first widely deployed SIMD on desktop CPUs, followed by more advanced instruction sets like SSE, AVX, and AVX-512.

[^flynn-1966]:  Flynn, Michael J. (December 1966). "Very high-speed computing systems" (PDF). Proceedings of the IEEE. 54 (12): 1901–1909.

SIMD became standard in most modern CPUs, accelerating tasks such as digital image processing, audio processing, and gaming graphics by executing the same operation on multiple data points at once. Thus, SIMD evolved from early experimental supercomputers to an integral part of everyday computing.

## SIMD on Apple Silicon

Apple Silicon processors, including the M1, M2, and M3 series, primarily support the [ARM NEON](https://developer.arm.com/Architectures/Neon) instruction set, which is a 128-bit SIMD architecture part of the ARMv8-A ISA[^armv8]. NEON provides a wide range of integer and floating-point vector operations suitable for parallel processing on vectors of data.

[^armv8]: The ARMv8-A ISA (Instruction Set Architecture) is a 64-bit architecture developed by ARM Holdings, supporting both 32-bit (AArch32) and 64-bit (AArch64) execution states. It includes three main instruction sets: A32 and T32 for 32-bit processing, and A64 for 64-bit processing. An ISA is an abstract model that defines how software controls a processor, specifying the set of machine-level instructions the CPU can execute, along with how they are encoded and how they interact with registers and memory.

In addition to NEON, Apple Silicon also features proprietary Apple Matrix Coprocessor (AMX)[^apple-amx] instructions. These instructions are specialized for high-performance computing tasks involving matrix operations and are unique to Apple Silicon. They are not part of the official ARM architecture and are currently undocumented publicly[^amx-docs], but they add significant computational acceleration beyond NEON for certain workloads.

[^apple-amx]: Apple's AMX is not to be confused with Intel's AMX.

The best way to utilize AMX instructions is through the Accelerate framework, which provides a high-level API for performing BLAS operations.

[^amx-docs]: You can find some [user-written documentation](https://github.com/corsix/amx) on GitHub.

## OpenBLAS vs Accelerate

There are several CPU-based BLAS libraries, some of them developed by hardware manifacturers, such as [Intel's MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html), [AMD's BLIS](https://www.amd.com/de/developer/aocl/blis.html) and [Apple's Accelerate](https://developer.apple.com/documentation/accelerate). OpenBLAS is an open source library that has the broadest coverage of supported hardware and can be a solid default choice.

I was curious what the difference would be in terms of performance when comparing OpenBLAS and Accelerate on Apple Silicon. Since OpenBLAS does not support Apple's AMX instructions, relying on NEON instructions only, I expect it to be slower for most use cases.

I built four benchmarks around some common double precision BLAS routines that are part of the C BLAS interface:

- `cblas_daxpy` – *a · X + Y* is a level 1 routine that scales vector *X* by scalar *a* and adds it to vector *Y*. This routine is used in linear combination of vectors, common in iterative algorithms like conjugate gradient (CG) or generalized minimum residual (GMRES) for solving linear systems.
- `cblas_ddot` - *X · Y* is a level 1 routine that computes the dot product of vectors *X* and *Y*. Dot products are widely used in machine learning, e.g. to compute similarities between vectors or phsysics simulations.
- `cblas_dgemv` - _a · **A** · x + b · y_ is a level 2 routine that handles matrix-vector multiplication. _a_, _b_ are scalars, *x*, *y* are vectors, and _**A**_ is a matrix. This routine is also common in iterative algorithms, but also in machine learning (linear regression), and signal processing.
- `cblas_dgemm` - _a · **A** · **B** + b · **C**_ is a level 3 routine that handles matrix-matrix multiplication. _a_, _b_ are scalars, and _**A**_, _**B**_, _**C**_ are matrices. This routine is common in machine learning and graphics processing, for example.

The benchmarks use Google's [benchmark](https://github.com/google/benchmark) library to measure performance. You can find the full code on [GitHub](https://github.com/frosnerd/cpp-simd-post). The following listing outlines the benchmark for `cblas_ddot` defined as a macro taking the name of the BLAS backend as an argument:

```c++
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
```

We'll obtain the desired vector size `n` from the benchmark state. We'll initialize two vectors of the same size `x` and `y`. The `for (auto _ : state)` loop runs the function for the desired number of iterations. `benchmark::DoNotOptimize` is used to prevent the compiler from optimizing away the function call because the result is unused. We'll record the user metric number of items processed as the number of iterations times the vector size.

We'll register the function as a benchmark using the `BENCHMARK` macro, defining the range of vector sizes to test, e.g. from 2<sup>2</sup> to 2<sup>22</sup> with a multiplier of 2. We can then generate the benchmarks by calling the macro with the desired backend name:

```c++
#ifdef USE_ACCELERATE
BENCHMARK_DDOT(Accelerate)
BENCHMARK_DGEMM(Accelerate)
BENCHMARK_DGEMV(Accelerate)
BENCHMARK_DAXPY(Accelerate)
#endif
```

We'll repeat the same for OpenBLAS. In our `CMakeLists.txt` file we can then conditionally compile the two versions, or simply compile both versions at once:

```cmake
# ...
option(BUILD_ACCELERATE "Build binary with Apple Accelerate framework (macOS only)" ON)
option(BUILD_OPENBLAS "Build binary with OpenBLAS" ON)
# ...
if(BUILD_ACCELERATE AND APPLE)
    add_executable(cpp-simd-post-accelerate ${SOURCES})
    target_link_libraries(cpp-simd-post-accelerate PRIVATE benchmark::benchmark benchmark::benchmark_main)
    target_link_libraries(cpp-simd-post-accelerate PRIVATE "-framework Accelerate")
    target_compile_definitions(cpp-simd-post-accelerate PRIVATE ACCELERATE_NEW_LAPACK USE_ACCELERATE)
    message(STATUS "Building cpp-simd-post-accelerate with Apple Accelerate framework")
endif()
# ...
```

When checking the resulting binaries, we can indeed see that they link to the respective libraries (assuming you installed them correctly on your system first):

```
$ otool -fahl build/cpp-simd-post-openblas | grep openblas -B5 -A5

Load command 14
          cmd LC_LOAD_DYLIB
      cmdsize 80
         name /opt/homebrew/opt/openblas/lib/libopenblas.0.dylib (offset 24)
   time stamp 2 Thu Jan  1 01:00:02 1970
      current version 0.0.0
compatibility version 0.0.0
```

```
$ otool -fahl build/cpp-simd-post-accelerate | grep Accelerate -B5 -A5

Load command 14
          cmd LC_LOAD_DYLIB
      cmdsize 96
         name /System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate (offset 24)
   time stamp 2 Thu Jan  1 01:00:02 1970
      current version 4.0.0
compatibility version 1.0.0
```

## Results

```bash
./build/cpp-simd-post-openblas --benchmark_out="openblas.json"
./build/cpp-simd-post-accelerate --benchmark_out="accelerate.json"
```

![daxpy benchmark results](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/sn4ypnz8yi5hv6pq09hc.png)
![ddot benchmark results](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/lrb02axyd04mqpzadivl.png)
![dgemm benchmark results](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/kijbe9iviaig6o9g1n4s.png)
![dgemv benchmark results](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/u5j972ba5o6i4dy28hy0.png)

```
OPENBLAS_VERBOSE=2 ./build/cpp-simd-post-openblas
Core: neoversen1
```

