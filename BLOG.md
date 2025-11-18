---
title: Comparing OpenBLAS and Accelerate on Apple Silicon for BLAS Routines
published: true
description: In this post we'll write and run benchmarks for four common BLAS routines, comparing two different BLAS implementations in C++.
tags: cpp, simd, matrix, cpu
cover_image: https://dev-to-uploads.s3.amazonaws.com/uploads/articles/q10ektung4q5rmcei9su.png
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

[^amx-docs]: You can find some [user-written documentation and header files](https://github.com/corsix/amx) on GitHub. There's also [The Elusive Apple Matrix Coprocessor](https://research.meekolab.com/the-elusive-apple-matrix-coprocessor-amx) blog post by Meeko Labs which is worth reading.

## OpenBLAS vs Accelerate

### Benchmark Setup

There are several CPU-based BLAS libraries, some of them developed by hardware manifacturers, such as [Intel's MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html), [AMD's BLIS](https://www.amd.com/de/developer/aocl/blis.html) and [Apple's Accelerate](https://developer.apple.com/documentation/accelerate). [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) is an open source library that has the broadest coverage of supported hardware and can be a solid default choice.

I was curious what the difference would be in terms of performance when comparing OpenBLAS and Accelerate on Apple Silicon. Since OpenBLAS does not support Apple's AMX instructions, relying on NEON instructions only, I expect it to be slower for most use cases.

I built four benchmarks around some common double precision BLAS routines that are part of the C BLAS interface:

- `cblas_daxpy` – *a · X + Y* is a level 1 routine that scales vector *X* by scalar *a* and adds it to vector *Y*. This routine is used in linear combination of vectors, common in iterative algorithms like conjugate gradient (CG) or generalized minimum residual (GMRES) for solving linear systems.
- `cblas_ddot` - *X · Y* is a level 1 routine that computes the dot product of vectors *X* and *Y*. Dot products are widely used in machine learning, e.g. to compute similarities between vectors or physics simulations.
- `cblas_dgemv` - _a · **A** · X + b · X_ is a level 2 routine that handles matrix-vector multiplication. _a_, _b_ are scalars, *X*, *Y* are vectors, and _**A**_ is a matrix. This routine is also common in iterative algorithms, but also in machine learning (linear regression), and signal processing.
- `cblas_dgemm` - _a · **A** · **B** + b · **C**_ is a level 3 routine that handles matrix-matrix multiplication. _a_, _b_ are scalars, and _**A**_, _**B**_, _**C**_ are matrices. This routine is common in machine learning and graphics processing, for example.

The benchmarks use Google's [benchmark](https://github.com/google/benchmark) library to measure performance. You can find the full code on [GitHub](https://github.com/frosnerd/cpp-simd-post). The following listing outlines the benchmark for `cblas_ddot` defined as a macro taking the name of the BLAS backend as an argument:

```cpp
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

```cpp
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

Please note that OpenBLAS was using the `neoversen1` core on my Apple M3 Pro chip.

```
OPENBLAS_VERBOSE=2 ./build/cpp-simd-post-openblas
Core: neoversen1
```

### Generating the Results

Now that we have our benchmarks compiled, let's run them and analyze the results. We can use the `--benchmark_out` command line argument to specify a JSON output file that will hold the results.

```bash
./build/cpp-simd-post-openblas --benchmark_out="openblas.json"
./build/cpp-simd-post-accelerate --benchmark_out="accelerate.json"
```

Aside from some metadata about the run, the JSON file contains a list of benchmark results of the following form:

```json
{
  "name": "BM_Ddot_Accelerate/2",
  "family_index": 0,
  "per_family_instance_index": 0,
  "run_name": "BM_Ddot_Accelerate/2",
  "run_type": "iteration",
  "repetitions": 1,
  "repetition_index": 0,
  "threads": 1,
  "iterations": 76769538,
  "real_time": 9.1229957239181410e+00,
  "cpu_time": 9.1226027698642671e+00,
  "time_unit": "ns",
  "items_per_second": 2.1923567762994435e+08
}
```

The `name` field encodes the benchmark name (`Ddot`), the library (`Accelerate`), and the input size (`2`). For our analysis we will look at the user metric `items_per_second` (larger is better), which represents the number of input doubles we were able to process per second. I wrote a Python script that collects those benchmark results, parses the `name` field and plots the results. Let's take a look at them.

### Analyzing the Results

Let's start with `cblas_daxpy`, which scales a vector and adds it to another vector. We'll benchmark vector sizes from 2 to 4,194,304. While most applications rely on smaller vectors up to 2<sup>12</sup> =  4096  elements, larger vectors are not unheard of.

![daxpy benchmark results](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/sn4ypnz8yi5hv6pq09hc.png)

We can see that in very small vector sizes, the performance difference is insignificant, and OpenBLAS even outperforms Accelerate. However, starting with input size 512, Accelerate surpasses OpenBLAS and remains consistently better by a factor of up to 6x, especially because OpenBLAS seems to slow down significantly starting from 2<sup>14</sup> input size. What's interesting is that OpenBLAS picks up the pace again and surpasses Accelerate for input sizes larger than 2<sup>18</sup>. 

Now let's look at the `cblas_ddot` results. We'll use the same input sizes to benchmark.

![ddot benchmark results](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/lrb02axyd04mqpzadivl.png)

Unsurprisingly, the results are very similar, with the performance being comparable for very small vector sizes, Accelerate outperforming OpenBLAS for medium to large sizes, and eventually OpenBLAS catching up for very large sizes. OpenBLAS also shows a dip in larger sizes, which looks interesting. We will investigate that later.

Next, let's dive into the matrix operations. For matrices, we'll use smaller input sizes (up to 2<sup>13</sup>), as the number of elements will be the squared input size. Typical input sizes in real world applications range from smaller matrices up to very large ones. Note that many matrix operations, especially for graphics processing and generative AI are offloaded to GPUs. Let's start with `cblas_dgemv`.

![dgemv benchmark results](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/u5j972ba5o6i4dy28hy0.png)

Similarly to the vector results, small sizes show little difference in performance. However, Accelerate starts to outperform OpenBLAS quite early. Even when both implementations have their peak performance (2<sup>10</sup>), Accelerate outperforms OpenBLAS. However, with growing matrix sizes, OpenBLAS takes the upper hand.

Let's check out `cblas_dgemm` next.

![dgemm benchmark results](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/kijbe9iviaig6o9g1n4s.png)

The `dgemm` results are the most consistent ones across all four experiments. Accelerate outperforms OpenBLAS for most input sizes up until 2<sup>13</sup>, from where OpenBLAS takes the lead. Interestingly, OpenBLAS does not show the dip in performance for larger input sizes as in the vector experiments. In `dgemm`, OpenBLAS also does not appear to have reached its peak performance in the given input range.

Looking at these results, we can note a few observations:

1. For very small input sizes, the performance difference is insignificant. I believe this is due to the fact that with small inputs, memory access and function call overhead dominate performance.
2. For medium to large input sizes, Accelerate outperforms OpenBLAS. This is expected given that Accelerate can take advantage of Apple's AMX instructions and the AMX coprocessor, while OpenBLAS relies on NEON instructions only.
3. For very large input sizes, OpenBLAS outperforms Accelerate. I did not expect this, and I am not entirely sure what the reason behind the difference is. I am suspecting that either Accelerate is not optimized for very large input sizes, as those are not typical in the applications that run on consumer devices such as Macs, iPhones, etc., or it has to do with size limitations of the AMX coprocessor.
4. OpenBLAS shows a dip in performance for medium to large input sizes, which is especially visible in `daxpy` and `ddot`. I think that this might be caused by OpenBLAS dynamically choosing the number of threads based on the vector size. Let's investigate this a bit further.

Our benchmark run did not specify any parallelism and left the choise to the library. Based on the [ARM64 kernel](https://github.com/OpenMathLib/OpenBLAS/blob/f6df9bebbb4259aa61ab5634c0f1269fb152cc0e/kernel/arm64/dot.c#L85-L102) that is used for the `ddot` experiment, it seems that the number of threads is indeed chosen based on the input size, but also based on the OpenBLAS core used.

```cpp
static inline int get_dot_optimal_nthreads(BLASLONG n) {
  int ncpu = num_cpu_avail(1);

#if defined(NEOVERSEV1) && !defined(COMPLEX) && !defined(BFLOAT16)
  return get_dot_optimal_nthreads_neoversev1(n, ncpu);
#elif defined(DYNAMIC_ARCH) && !defined(COMPLEX) && !defined(BFLOAT16)
  if (strcmp(gotoblas_corename(), "neoversev1") == 0) {
    return get_dot_optimal_nthreads_neoversev1(n, ncpu);
  }
#endif

  // Default case
  if (n <= 10000L)
    return 1;
  else
    return num_cpu_avail(1);
}
#endif
```

Since my OpenBLAS uses `neoversen1`, it should choose one thread for vectors <= 10k, and all available CPUs for vectors > 10k. We can see that this threshold (red vertical line) aligns with the change in the performance trend:

![neoversen1 thresholds](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/q10ektung4q5rmcei9su.png)

## Summary and Conclusion

In this post we wrote and executed benchmarks for four common BLAS routines using two different BLAS implementations: OpenBLAS and Apple's Accelerate framework. We used Google's C++ benchmark library to measure performance and plotted the results using Python. We found the following results:

- For very small input sizes, results of both libraries are comparable. This is likely due to the fact that with small inputs, memory access and function call overhead dominate performance. 
- Accelerate outperforms OpenBLAS for medium to large input sizes. This is expected given that Accelerate can take advantage of Apple's AMX instructions and the AMX coprocessor, while OpenBLAS relies on NEON instructions only. If there are specialized, state-of-the-art instructions available on your platform, you should use a BLAS library that takes advantage of them.
- While the BLAS interfaces are the same, the available configuration options, e.g. the parallelism, can vary between implementations and have a significant impact on the performance.
- OpenBLAS appears to outperform Accelerate for very large input sizes. There is no silver bullet.

Even when not rolling your own BLAS, relying on well-known BLAS libraries, there can be performance differences between the different implementations in orders of magnitude. If your application requires maximum performance, you should always benchmark the different options and choose the one that performs best for your use case on the hardware you run in production.

Note that during my benchmarks I only looked at throughput. Latency is another relevant metric to consider, especially for real-time applications. Latency can be significantly higher when using coprocessors or processing units farther away from the CPU, such as GPUs.

Have you compared different BLAS or scientific computing libraries before? Did you run into any unexplained performance differences? I'd love to hear about your experiences in the comments below!

---

If you liked this post, you can [support me on ko-fi](https://ko-fi.com/frosnerd).