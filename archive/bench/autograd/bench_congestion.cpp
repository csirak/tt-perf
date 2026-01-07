// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: Parallel MLP workloads for congestion testing
// Runs multiple MLPs concurrently to measure data parallel throughput.
// Tests if running N partitions gives Nx throughput (congestion ratio).
//
// Approach: Use async execution pattern - enqueue all ops, single Finish at end.
// This tests whether multiple MLPs can overlap execution on the same device.

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <tt-metalium/distributed.hpp>

#include <chrono>
#include <vector>
#include <numeric>

using namespace ttnn;
using namespace tt::tt_metal::distributed;

constexpr int N_WARMUP = 5;
constexpr int N_TIMED = 20;

struct SizeConfig {
    const char* name;
    uint32_t batch;
    uint32_t dim;
};

// MLP tensors for one instance
struct MLPInstance {
    Tensor x, w1, b1, w2, b2, dout;
};

// Create MLP instance tensors
MLPInstance create_mlp(MeshDevice& dev, uint32_t batch, uint32_t dim) {
    return {
        ones(Shape({batch, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        zeros(Shape({1, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        zeros(Shape({1, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        ones(Shape({batch, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
    };
}

// Run forward+backward for one MLP (non-blocking, enqueues ops)
void run_mlp_async(const MLPInstance& m, const std::optional<CoreGrid>& grid) {
    // Forward
    auto mm1 = ttnn::matmul(m.x, m.w1, false, true, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, std::nullopt, grid);
    auto h_pre = ttnn::add(mm1, m.b1);
    auto h = ttnn::relu(h_pre);
    auto mm2 = ttnn::matmul(h, m.w2, false, true, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, std::nullopt, grid);
    auto out = ttnn::add(mm2, m.b2);

    // Backward
    auto dh = ttnn::matmul(m.dout, m.w2, false, false, std::nullopt, std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, grid);
    auto relu_mask = ttnn::gtz(h_pre);
    auto dh_pre = ttnn::multiply(dh, relu_mask);
    auto dx = ttnn::matmul(dh_pre, m.w1, false, false, std::nullopt, std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, grid);
}

// Benchmark: Run N MLPs concurrently with interleaved ops
// This simulates data parallel where each MLP processes different data
double bench_parallel_mlps(MeshDevice& dev, const SizeConfig& size,
                           int n_parallel, const std::optional<CoreGrid>& grid) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    // Create N MLP instances
    std::vector<MLPInstance> mlps;
    for (int i = 0; i < n_parallel; i++) {
        mlps.push_back(create_mlp(dev, size.batch, size.dim));
    }
    Finish(cq);

    // Warmup - run all MLPs interleaved
    for (int w = 0; w < N_WARMUP; w++) {
        for (const auto& m : mlps) {
            run_mlp_async(m, grid);
        }
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        // Enqueue all MLPs (async - ops are pipelined)
        for (const auto& m : mlps) {
            run_mlp_async(m, grid);
        }
        Finish(cq);  // Wait for all to complete

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

// Benchmark: Solo MLP (baseline)
double bench_solo_mlp(MeshDevice& dev, const SizeConfig& size) {
    return bench_parallel_mlps(dev, size, 1, std::nullopt);
}

int main() {
    auto device = MeshDevice::create_unit_mesh(0);

    auto grid_size = device->compute_with_storage_grid_size();
    fmt::print("# Device core grid: {}x{} = {} cores\n", grid_size.x, grid_size.y,
               grid_size.x * grid_size.y);

    // Size configurations
    std::vector<SizeConfig> sizes = {
        {"small",  32,   256},
        {"medium", 256,  512},
        {"large",  512,  1024},
    };

    // Parallelism levels to test
    std::vector<int> parallel_counts = {1, 2, 4, 8};

    fmt::print("# Parallel MLP Congestion Benchmark\n");
    fmt::print("# Tests data parallel throughput with N concurrent MLPs\n");
    fmt::print("# Network: x[B,D] -> Linear+ReLU[D] -> Linear[D]\n");
    fmt::print("# Warmup: {}, Timed iterations: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");
    fmt::print("# Congestion ratio = total_throughput / (N × solo_throughput)\n");
    fmt::print("# Ratio = 1.0 means perfect scaling, <1.0 means congestion\n");
    fmt::print("#\n");
    fmt::print("size,batch,dim,n_parallel,total_ms,solo_ms,total_throughput,expected_throughput,congestion_ratio\n");

    for (const auto& size : sizes) {
        // First measure solo baseline
        double solo_ms = bench_solo_mlp(*device, size);
        double solo_throughput = size.batch / solo_ms * 1000.0;

        for (int n : parallel_counts) {
            double total_ms = bench_parallel_mlps(*device, size, n, std::nullopt);

            // Total throughput = N batches / time
            double total_throughput = (n * size.batch) / total_ms * 1000.0;
            double expected_throughput = n * solo_throughput;
            double congestion_ratio = total_throughput / expected_throughput;

            fmt::print("{},{},{},{},{:.3f},{:.3f},{:.1f},{:.1f},{:.3f}\n",
                       size.name, size.batch, size.dim, n,
                       total_ms, solo_ms, total_throughput, expected_throughput,
                       congestion_ratio);
        }
        fmt::print("#\n");
    }

    fmt::print("# Analysis:\n");
    fmt::print("# - Congestion ratio ~1.0: Operations fully overlap (memory-bound)\n");
    fmt::print("# - Congestion ratio <1.0: Shared resource contention (DRAM/NoC bandwidth)\n");
    fmt::print("# - Congestion ratio >1.0: Impossible, indicates measurement error\n");

    device->close();
    return 0;
}
