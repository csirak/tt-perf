// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Matmul sweep benchmark across multiple sizes using TTNN C++ API

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <tt-metalium/distributed.hpp>

#include <array>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>

using namespace ttnn;
using namespace tt::tt_metal::distributed;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;
using DispatchCoreType = tt::tt_metal::DispatchCoreType;

constexpr int N_WARMUP = 10;
constexpr int N_ITER = 50;

struct BenchResult {
    uint32_t size;
    double mean_us;
    double std_us;
    double min_us;
    double max_us;
};

BenchResult benchmark_matmul(MeshDevice& device, uint32_t size) {
    std::array<uint32_t, 2> shape = {size, size};

    auto a = ones(Shape(shape), DataType::BFLOAT16, TILE_LAYOUT, device);
    auto b = ones(Shape(shape), DataType::BFLOAT16, TILE_LAYOUT, device);

    // Get command queue for synchronization
    MeshCommandQueue& cq = device.mesh_command_queue();

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        auto warmup = ttnn::matmul(a, b);
        Finish(cq);  // Sync after each warmup
    }

    // Timed runs
    std::vector<double> times_us(N_ITER);

    for (int i = 0; i < N_ITER; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto c = ttnn::matmul(a, b);
        Finish(cq);  // Wait for device to complete
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

    return BenchResult{
        .size = size,
        .mean_us = mean,
        .std_us = std_dev,
        .min_us = times_us.front(),
        .max_us = times_us.back(),
    };
}

int main() {
    // Set USE_ETH_DISPATCH=1 to use ETH dispatch (8x8 grid) instead of WORKER (8x7 grid)
    bool use_eth = std::getenv("USE_ETH_DISPATCH") != nullptr;
    auto device = use_eth
        ? MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1,
            DispatchCoreConfig{DispatchCoreType::ETH})
        : open_mesh_device(0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE);

    auto grid = device->compute_with_storage_grid_size();
    fmt::print("# Compute grid: {}x{} = {} cores\n", grid.x, grid.y, grid.x * grid.y);

    std::vector<uint32_t> sizes = {128, 256, 512, 1024, 2048, 4096};
    std::vector<BenchResult> results;

    fmt::print("backend,size,mean_us,std_us,min_us,max_us\n");

    for (uint32_t size : sizes) {
        auto result = benchmark_matmul(*device, size);
        results.push_back(result);

        fmt::print("ttnn_cpp,{},{:.2f},{:.2f},{:.2f},{:.2f}\n",
                   result.size, result.mean_us, result.std_us,
                   result.min_us, result.max_us);
    }

    // Cleanup
    Finish(device->mesh_command_queue());
    ttnn::close_device(*device);

    return 0;
}
