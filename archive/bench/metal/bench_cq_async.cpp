// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: Command Queue Async vs Sync
// Demonstrates the benefit of non-blocking (async) command queue operations
// by comparing batched async dispatch vs per-op synchronization.

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <tt-metalium/distributed.hpp>

#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>

using namespace ttnn;
using namespace tt::tt_metal::distributed;

constexpr int N_WARMUP = 5;
constexpr int N_TIMED = 10;
constexpr uint32_t SIZE = 512;  // Matrix size (small to emphasize dispatch overhead)

// Sync benchmark: Finish() after each matmul
double bench_sync(MeshDevice& dev, int n_ops) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    auto a = ones(Shape({SIZE, SIZE}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto b = ones(Shape({SIZE, SIZE}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    Finish(cq);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        auto c = ttnn::matmul(a, b);
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_ms(N_TIMED);

    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_ops; i++) {
            auto c = ttnn::matmul(a, b);
            Finish(cq);  // Sync after each op
        }

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Return mean
    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

// Async benchmark: batch all matmuls, single Finish() at end
double bench_async(MeshDevice& dev, int n_ops) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    auto a = ones(Shape({SIZE, SIZE}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto b = ones(Shape({SIZE, SIZE}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    Finish(cq);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        auto c = ttnn::matmul(a, b);
    }
    Finish(cq);

    // Timed runs
    std::vector<double> times_ms(N_TIMED);

    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_ops; i++) {
            auto c = ttnn::matmul(a, b);  // Non-blocking enqueue
        }
        Finish(cq);  // Single sync at end

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Return mean
    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

int main() {
    auto device = MeshDevice::create_unit_mesh(0);

    fmt::print("# Command Queue Async vs Sync Benchmark\n");
    fmt::print("# Matrix size: {}x{}, Warmup: {}, Timed iterations: {}\n", SIZE, SIZE, N_WARMUP, N_TIMED);
    fmt::print("#\n");
    fmt::print("# Sync:  Finish() after each matmul (host waits for each op)\n");
    fmt::print("# Async: Batch all matmuls, single Finish() at end (pipelined)\n");
    fmt::print("#\n");
    fmt::print("n_ops,sync_ms,async_ms,speedup\n");

    std::vector<int> n_ops_list = {5, 10, 20, 50, 100};

    for (int n_ops : n_ops_list) {
        double sync_ms = bench_sync(*device, n_ops);
        double async_ms = bench_async(*device, n_ops);
        double speedup = sync_ms / async_ms;

        fmt::print("{},{:.3f},{:.3f},{:.2f}x\n", n_ops, sync_ms, async_ms, speedup);
    }

    device->close();
    return 0;
}
