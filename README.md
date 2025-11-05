# cpp-simd-post

## Building

### Default Build (Apple Accelerate on macOS)

```bash
cmake -S . -B build
cmake --build build
```

### Build with OpenBLAS

First, install OpenBLAS:
```bash
# macOS
brew install openblas

# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# Fedora/RHEL
sudo dnf install openblas-devel
```

Then build with OpenBLAS enabled:
```bash
cmake -S . -B build -DUSE_OPENBLAS=ON -DUSE_ACCELERATE=OFF
cmake --build build
```

**Note for macOS users**: OpenBLAS is keg-only in Homebrew and not symlinked by default. The CMake configuration automatically detects the Homebrew installation path. If CMake cannot find OpenBLAS, you can manually set the path:
```bash
export CMAKE_PREFIX_PATH="/opt/homebrew/opt/openblas"
cmake -S . -B build -DUSE_OPENBLAS=ON -DUSE_ACCELERATE=OFF
cmake --build build
```

### Build Options

- `USE_ACCELERATE`: Use Apple Accelerate framework (default: ON on macOS, requires macOS)
- `USE_OPENBLAS`: Use OpenBLAS (default: OFF)

You can enable both to compare performance between the two implementations.

## Running

```bash
./build/cpp-simd-post
```
