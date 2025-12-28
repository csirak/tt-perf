// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// TRUE Tensor-Parallel Matmul on 2-device N300 (1x2 mesh)
// Column-parallel pattern:
//   - Replicate A to all devices
//   - Shard B by columns (dim=1) across devices
//   - Each device computes: A × B_shard = C_shard
//   - all_gather to combine results
//
// This HALVES the computation per device (actual speedup expected)

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/ccl/all_gather/all_gather.hpp>
#include <ttnn/distributed/api.hpp>
#include <ttnn/distributed/types.hpp>
#include <ttnn/distributed/distributed_tensor.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_config.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

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

BenchResult benchmark_tensor_parallel_matmul(MeshDevice& mesh_device, uint32_t size, int num_devices) {
    // Create HOST tensors using Tensor::from_vector (required format for distribute_tensor)
    tt::tt_metal::TensorSpec tensor_spec(
        ttnn::Shape({size, size}),
        tt::tt_metal::TensorLayout(DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::TILE), tt::tt_metal::MemoryConfig{})
    );

    // Create data vectors filled with 1.0
    size_t num_elements = static_cast<size_t>(size) * size;
    std::vector<bfloat16> a_data(num_elements, bfloat16(1.0f));
    std::vector<bfloat16> b_data(num_elements, bfloat16(1.0f));

    auto a_host = tt::tt_metal::Tensor::from_vector(std::move(a_data), tensor_spec);
    auto b_host = tt::tt_metal::Tensor::from_vector(std::move(b_data), tensor_spec);

    // Create mappers
    auto replicate_mapper = ttnn::distributed::replicate_tensor_to_mesh_mapper(mesh_device);
    auto shard_mapper = ttnn::distributed::shard_tensor_to_mesh_mapper(mesh_device, 1);  // shard dim=1 (columns)

    // Distribute: A replicated, B sharded by columns
    auto a = ttnn::distributed::distribute_tensor(a_host, *replicate_mapper, std::ref(mesh_device));
    auto b = ttnn::distributed::distribute_tensor(b_host, *shard_mapper, std::ref(mesh_device));

    MeshCommandQueue& cq = mesh_device.mesh_command_queue();

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        auto c_sharded = ttnn::matmul(a, b);  // Each device: N×N × N×(N/2) = N×(N/2)
        auto c_full = ttnn::all_gather(c_sharded, 1);  // Gather along dim=1
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_us(N_ITER);

    for (int i = 0; i < N_ITER; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto c_sharded = ttnn::matmul(a, b);
        auto c_full = ttnn::all_gather(c_sharded, 1);
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

    // TFLOP/s: 2 * N^3 / time (total work is same as single device)
    // The speedup comes from TIME reduction, not increased FLOPS
    double flops = 2.0 * std::pow(size, 3);
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

    // Powers of 2 (must be divisible by 2 for column sharding)
    for (int exp = 6; exp <= 12; exp++) {  // 64 to 4096 (skip 32, not enough columns)
        sizes.insert(1 << exp);
    }

    // Fine sweep around key points (all divisible by 64 for clean sharding)
    struct Region { uint32_t start, end, step; };
    std::vector<Region> regions = {
        {256, 512, 64},
        {896, 1152, 64},
        {1920, 2176, 64},
        {3840, 4096, 64},
    };

    for (const auto& r : regions) {
        for (uint32_t s = r.start; s <= r.end; s += r.step) {
            if (s % 64 == 0) {  // Must be divisible by 64 (32 tile × 2 devices)
                sizes.insert(s);
            }
        }
    }

    return std::vector<uint32_t>(sizes.begin(), sizes.end());
}

int main() {
    // Enable fabric for CCL operations (all_gather requires this)
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);

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

    fmt::print("# Tensor-Parallel Matmul on 1x2 MeshDevice (N300)\n");
    fmt::print("# {} devices with column-parallel sharding\n", num_devices);
    fmt::print("# Pattern: A (replicated) × B (col-sharded) → C (col-sharded) → all_gather\n");
    fmt::print("size,mean_us,std_us,min_us,max_us,tflops\n");

    for (uint32_t size : sizes) {
        try {
            auto result = benchmark_tensor_parallel_matmul(*mesh_device, size, num_devices);
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
