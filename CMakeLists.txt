cmake_minimum_required(VERSION 3.31)
project(cpp-simd-post)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
if(APPLE)
    target_link_libraries(cpp-simd-post PRIVATE "-framework Accelerate")
    target_compile_definitions(cpp-simd-post PRIVATE ACCELERATE_NEW_LAPACK)
endif()
