// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Distributed matmul benchmark on 2-device N300 (1x2 mesh)
// Both devices run the same matmul in parallel (data parallel)
// This measures the overhead of mesh device vs single device

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/distributed/types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/dispatch_core_common.hpp>

#include <array>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <set>

using namespace ttnn;
using namespace tt::tt_metal::distributed;

constexpr int N_WARMUP = 5;
constexpr int N_ITER = 20;

struct BenchResult {
    uint32_t size;
    double mean_us;
    double std_us;
    double min_us;
    double max_us;
    double tflops;
};

BenchResult benchmark_mesh_matmul(MeshDevice& mesh_device, uint32_t size, int num_devices) {
    std::array<uint32_t, 2> shape = {size, size};

    // Create tensors on mesh device (replicated to all devices)
    auto a = ones(Shape(shape), DataType::BFLOAT16, TILE_LAYOUT, mesh_device);
    auto b = ones(Shape(shape), DataType::BFLOAT16, TILE_LAYOUT, mesh_device);

    MeshCommandQueue& cq = mesh_device.mesh_command_queue();

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        auto c = ttnn::matmul(a, b);
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_us(N_ITER);

    for (int i = 0; i < N_ITER; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto c = ttnn::matmul(a, b);
        Finish(cq);
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

    // TFLOP/s: num_devices * 2 * N^3 / time (each device does full matmul)
    double flops = static_cast<double>(num_devices) * 2.0 * std::pow(size, 3);
    double tflops = flops / (mean * 1e-6) / 1e12;

    return BenchResult{
        .size = size,
        .mean_us = mean,
        .std_us = std_dev,
        .min_us = times_us.front(),
        .max_us = times_us.back(),
        .tflops = tflops,
    };
}

std::vector<uint32_t> generate_test_sizes() {
    std::set<uint32_t> sizes;

    // Powers of 2
    for (int exp = 5; exp <= 12; exp++) {  // 32 to 4096
        sizes.insert(1 << exp);
    }

    // Fine sweep around inflection points
    struct Region { uint32_t start, end, step; };
    std::vector<Region> regions = {
        {192, 384, 32},
        {384, 640, 32},
        {896, 1152, 32},
        {1920, 2176, 32},
        {3840, 4096, 64},
    };

    for (const auto& r : regions) {
        for (uint32_t s = r.start; s <= r.end; s += r.step) {
            if (s % 32 == 0) {
                sizes.insert(s);
            }
        }
    }

    return std::vector<uint32_t>(sizes.begin(), sizes.end());
}

int main() {
    // Open 1x2 mesh device (2 devices)
    auto mesh_device = ttnn::distributed::open_mesh_device(
        MeshShape(1, 2),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        1,  // num_command_queues
        tt::tt_metal::DispatchCoreConfig{}
    );

    int num_devices = mesh_device->num_devices();
    auto sizes = generate_test_sizes();

    fmt::print("# Mesh matmul benchmark on 1x2 MeshDevice (N300)\n");
    fmt::print("# {} devices running matmul in parallel (data parallel)\n", num_devices);
    fmt::print("size,mean_us,std_us,min_us,max_us,tflops\n");

    for (uint32_t size : sizes) {
        try {
            auto result = benchmark_mesh_matmul(*mesh_device, size, num_devices);
            fmt::print("{},{:.2f},{:.2f},{:.2f},{:.2f},{:.3f}\n",
                       result.size, result.mean_us, result.std_us,
                       result.min_us, result.max_us, result.tflops);
        } catch (const std::exception& e) {
            fmt::print(stderr, "# Error at size {}: {}\n", size, e.what());
        }
    }

    ttnn::distributed::close_mesh_device(mesh_device);
    return 0;
}
