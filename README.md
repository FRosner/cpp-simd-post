# cpp-simd-post

## Building

### Prerequisites

For OpenBLAS support, install OpenBLAS:
```bash
# macOS
brew install openblas

# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# Fedora/RHEL
sudo dnf install openblas-devel
```

### Default Build (Both Accelerate and OpenBLAS on macOS)

By default, both binaries are built:

```bash
cmake -S . -B build
cmake --build build
```

This creates two executables:
- `build/cpp-simd-post-accelerate` - Uses Apple Accelerate framework
- `build/cpp-simd-post-openblas` - Uses OpenBLAS

## Running

Run the Accelerate version:
```bash
./build/cpp-simd-post-accelerate
```

Run the OpenBLAS version:
```bash
./build/cpp-simd-post-openblas
```

This allows you to easily compare performance between the two BLAS implementations.

## Benchmark Analysis

```bash
./build/cpp-simd-post-openblas --benchmark_out="openblas.json"
./build/cpp-simd-post-accelerate --benchmark_out="accelerate.json"
python3 benchmark_analysis.py
```

This will generate plots in `plots`.

## Debugging

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

```bash
lldb build/cpp-simd-post-accelerate

break set -n blas_ddot
run
step
```
