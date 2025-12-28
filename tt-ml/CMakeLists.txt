# tt-ml: Simple training test using tt-train's TTML library
# This CMakeLists.txt is designed to be included as a subdirectory in tt-train

cmake_minimum_required(VERSION 3.18)
project(tt-ml LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# simple_mlp executable - links against ttml from tt-train
add_executable(simple_mlp src/simple_mlp.cpp)
target_link_libraries(simple_mlp PRIVATE ttml)

# bench_sweep executable - scaling sweep benchmark
add_executable(bench_sweep src/bench_sweep.cpp)
target_link_libraries(bench_sweep PRIVATE ttml)

# bench_sweep_traced executable - non-traced + traced sweep benchmark
add_executable(bench_sweep_traced src/bench_sweep_traced.cpp)
target_link_libraries(bench_sweep_traced PRIVATE ttml)
