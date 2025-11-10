---
title: Using Apple's Accelerate Framework for BLAS Operations
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

## Results

```bash
./build/cpp-simd-post-openblas --benchmark_out="openblas.json"
./build/cpp-simd-post-accelerate --benchmark_out="accelerate.json"
```


```
Benchmark                        Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------
BM_DdotOpenBLAS/1024           128 ns          128 ns      5475252 items_per_second=8.02053G/s
BM_DdotOpenBLAS/2048           241 ns          241 ns      2883803 items_per_second=8.50361G/s
BM_DdotOpenBLAS/4096           472 ns          472 ns      1477345 items_per_second=8.68423G/s
BM_DdotOpenBLAS/8192          1181 ns         1181 ns       591661 items_per_second=6.93466G/s
BM_DdotOpenBLAS/16384        45160 ns        14710 ns        45150 items_per_second=1.11381G/s
BM_DdotOpenBLAS/32768        45610 ns        14417 ns        48982 items_per_second=2.27284G/s
BM_DdotOpenBLAS/65536        46484 ns        14282 ns        47510 items_per_second=4.58866G/s
BM_DdotOpenBLAS/131072       47354 ns        14556 ns        47857 items_per_second=9.00479G/s
BM_DdotOpenBLAS/262144       51648 ns        17036 ns        39862 items_per_second=15.3875G/s
BM_DdotOpenBLAS/524288       60265 ns        21181 ns        32832 items_per_second=24.7527G/s
BM_DdotOpenBLAS/1048576      88170 ns        32628 ns        22805 items_per_second=32.1369G/s
BM_DdotOpenBLAS/2097152     274591 ns       159403 ns         4416 items_per_second=13.1563G/s
BM_DdotOpenBLAS/4194304     629821 ns       454093 ns         1535 items_per_second=9.23666G/s
BM_DgemmOpenBLAS/64          19608 ns        19607 ns        35895 bytes_per_second=4.66942Gi/s items_per_second=26.74G/s
BM_DgemmOpenBLAS/128        129020 ns        75756 ns         9312 bytes_per_second=4.83411Gi/s items_per_second=55.3663G/s
BM_DgemmOpenBLAS/256        344133 ns       216261 ns         2587 bytes_per_second=6.77351Gi/s items_per_second=155.157G/s
BM_DgemmOpenBLAS/512       1589663 ns      1212768 ns          638 bytes_per_second=4.83141Gi/s items_per_second=221.341G/s
BM_DgemmOpenBLAS/1024     10371197 ns      9089042 ns           71 bytes_per_second=2.57865Gi/s items_per_second=236.272G/s
BM_DgemmOpenBLAS/2048     73673349 ns     63063818 ns           11 bytes_per_second=1.48659Gi/s items_per_second=272.42G/s
```

```
------------------------------------------------------------------------------------
Benchmark                          Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------
BM_DdotAccelerate/1024           109 ns          109 ns      6394972 items_per_second=9.42305G/s
BM_DdotAccelerate/2048           125 ns          125 ns      5571252 items_per_second=16.3858G/s
BM_DdotAccelerate/4096           168 ns          168 ns      4165774 items_per_second=24.3713G/s
BM_DdotAccelerate/8192           246 ns          246 ns      2848632 items_per_second=33.3256G/s
BM_DdotAccelerate/16384          408 ns          408 ns      1728165 items_per_second=40.1663G/s
BM_DdotAccelerate/32768          725 ns          725 ns       934854 items_per_second=45.1869G/s
BM_DdotAccelerate/65536         1360 ns         1360 ns       515274 items_per_second=48.1733G/s
BM_DdotAccelerate/131072        2637 ns         2637 ns       265534 items_per_second=49.7119G/s
BM_DdotAccelerate/262144        5279 ns         5270 ns       133399 items_per_second=49.7464G/s
BM_DdotAccelerate/524288       10347 ns        10347 ns        67622 items_per_second=50.6702G/s
BM_DdotAccelerate/1048576      51355 ns        51354 ns        13967 items_per_second=20.4185G/s
BM_DdotAccelerate/2097152     326969 ns       326947 ns         2163 items_per_second=6.41434G/s
BM_DdotAccelerate/4194304     691531 ns       689796 ns         1069 items_per_second=6.0805G/s
BM_DgemmAccelerate/64           1628 ns         1616 ns       431287 bytes_per_second=56.659Gi/s items_per_second=324.465G/s
BM_DgemmAccelerate/128         10633 ns        10633 ns        66063 bytes_per_second=34.4409Gi/s items_per_second=394.46G/s
BM_DgemmAccelerate/256         79396 ns        79393 ns         8882 bytes_per_second=18.4504Gi/s items_per_second=422.635G/s
BM_DgemmAccelerate/512        664003 ns       663994 ns         1043 bytes_per_second=8.82444Gi/s items_per_second=404.274G/s
BM_DgemmAccelerate/1024      5498127 ns      5498128 ns          125 bytes_per_second=4.26281Gi/s items_per_second=390.585G/s
BM_DgemmAccelerate/2048     46171703 ns     46171667 ns           15 bytes_per_second=2.03047Gi/s items_per_second=372.087G/s

```

```
# otool -fahl build/cpp-simd-post-openblas | grep openblas -B5 -A5
Load command 14
          cmd LC_LOAD_DYLIB
      cmdsize 80
         name /opt/homebrew/opt/openblas/lib/libopenblas.0.dylib (offset 24)
   time stamp 2 Thu Jan  1 01:00:02 1970
      current version 0.0.0
compatibility version 0.0.0
```

```
# otool -fahl build/cpp-simd-post-accelerate | grep Accelerate -B5 -A5

Load command 14
          cmd LC_LOAD_DYLIB
      cmdsize 96
         name /System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate (offset 24)
   time stamp 2 Thu Jan  1 01:00:02 1970
      current version 4.0.0
compatibility version 1.0.0
```