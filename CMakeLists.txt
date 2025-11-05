cmake_minimum_required(VERSION 3.31)
project(cpp-simd-post)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_ACCELERATE "Use Apple Accelerate framework (macOS only)" ON)
option(USE_OPENBLAS "Use OpenBLAS" OFF)

# Collect source files (BLAS implementations are now in main.cpp)
file(GLOB SOURCES CONFIGURE_DEPENDS "*.cpp")

add_executable(cpp-simd-post ${SOURCES})

# Google Benchmark
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "" FORCE)
FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.4  # or the desired release tag
)
FetchContent_MakeAvailable(benchmark)
# Then link your target against benchmark::benchmark and benchmark::benchmark_main if needed?
target_link_libraries(cpp-simd-post PRIVATE benchmark::benchmark benchmark::benchmark_main)

# Link Accelerate framework on macOS for AMX support
if(USE_ACCELERATE AND APPLE)
    target_link_libraries(cpp-simd-post PRIVATE "-framework Accelerate")
    target_compile_definitions(cpp-simd-post PRIVATE ACCELERATE_NEW_LAPACK USE_ACCELERATE)
    message(STATUS "Using Apple Accelerate framework for BLAS")
endif()

# OpenBLAS
if(USE_OPENBLAS)
    # On macOS with Homebrew, OpenBLAS is keg-only and not symlinked
    if(APPLE)
        # Try to find Homebrew's OpenBLAS installation
        execute_process(
            COMMAND brew --prefix openblas
            OUTPUT_VARIABLE OPENBLAS_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(OPENBLAS_PREFIX)
            list(APPEND CMAKE_PREFIX_PATH "${OPENBLAS_PREFIX}")
            set(OPENBLAS_INCLUDE_DIR "${OPENBLAS_PREFIX}/include")
            set(OPENBLAS_LIB_DIR "${OPENBLAS_PREFIX}/lib")
            message(STATUS "Found Homebrew OpenBLAS at: ${OPENBLAS_PREFIX}")
        endif()
    endif()

    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS REQUIRED)
    target_link_libraries(cpp-simd-post PRIVATE ${BLAS_LIBRARIES})

    # Explicitly add OpenBLAS include directory
    if(OPENBLAS_INCLUDE_DIR)
        target_include_directories(cpp-simd-post PRIVATE ${OPENBLAS_INCLUDE_DIR})
        message(STATUS "OpenBLAS include directory: ${OPENBLAS_INCLUDE_DIR}")
    elseif(BLAS_INCLUDE_DIRS)
        target_include_directories(cpp-simd-post PRIVATE ${BLAS_INCLUDE_DIRS})
    endif()

    target_compile_definitions(cpp-simd-post PRIVATE USE_OPENBLAS)
    message(STATUS "Using OpenBLAS for BLAS")
endif()
