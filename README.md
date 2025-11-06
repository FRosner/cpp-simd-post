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

**Note for macOS users**: OpenBLAS is keg-only in Homebrew and not symlinked by default. The CMake configuration automatically detects the Homebrew installation path. If CMake cannot find OpenBLAS, you can manually set the path:
```bash
export CMAKE_PREFIX_PATH="/opt/homebrew/opt/openblas"
cmake -S . -B build
cmake --build build
```

### Build Options

- `BUILD_ACCELERATE`: Build binary with Apple Accelerate framework (default: ON, requires macOS)
- `BUILD_OPENBLAS`: Build binary with OpenBLAS (default: ON)

**Build only the Accelerate version:**
```bash
cmake -S . -B build -DBUILD_OPENBLAS=OFF
cmake --build build
```

**Build only the OpenBLAS version:**
```bash
cmake -S . -B build -DBUILD_ACCELERATE=OFF
cmake --build build
```

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
