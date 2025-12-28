// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Simple matmul using TTNN C++ API with timing

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>

#include <array>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace ttnn;

int main() {
    auto device = open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);

    // A[M,K] @ B[K,N] -> C[M,N]
    std::array<uint32_t, 2> shape_a = {512, 1024};
    std::array<uint32_t, 2> shape_b = {1024, 512};

    auto a = ones(Shape(shape_a), DataType::BFLOAT16, TILE_LAYOUT, *device);
    auto b = ones(Shape(shape_b), DataType::BFLOAT16, TILE_LAYOUT, *device);

    // Warmup (10 iterations)
    for (int i = 0; i < 10; i++) {
        auto warmup = ttnn::matmul(a, b);
    }

    // Timed runs (100 iterations)
    constexpr int N_ITER = 100;
    std::vector<double> times_us(N_ITER);

    for (int i = 0; i < N_ITER; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto c = ttnn::matmul(a, b);
        auto end = std::chrono::high_resolution_clock::now();
        times_us[i] = std::chrono::duration<double, std::micro>(end - start).count();
    }

    // Stats
    double sum = std::accumulate(times_us.begin(), times_us.end(), 0.0);
    double mean = sum / N_ITER;

    double sq_sum = 0.0;
    for (double t : times_us) {
        sq_sum += (t - mean) * (t - mean);
    }
    double std_dev = std::sqrt(sq_sum / N_ITER);

    std::sort(times_us.begin(), times_us.end());
    double min_time = times_us.front();
    double max_time = times_us.back();
    double median = times_us[N_ITER / 2];

    fmt::print("TTNN Matmul: [512x1024] @ [1024x512] -> [512x512]\n");
    fmt::print("Iterations: {}\n", N_ITER);
    fmt::print("Mean:   {:.2f} us ± {:.2f} us\n", mean, std_dev);
    fmt::print("Median: {:.2f} us\n", median);
    fmt::print("Min:    {:.2f} us\n", min_time);
    fmt::print("Max:    {:.2f} us\n", max_time);

    return 0;
}
