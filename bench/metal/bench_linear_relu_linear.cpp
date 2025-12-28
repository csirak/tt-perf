// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: Command Queue Sync vs Async for Linear-ReLU-Linear
// Compares per-op synchronization vs batched async dispatch across 3 sizes.

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <tt-metalium/distributed.hpp>

#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>

using namespace ttnn;
using namespace tt::tt_metal::distributed;

constexpr int N_WARMUP = 10;
constexpr int N_TIMED = 50;

struct BenchConfig {
    const char* name;
    uint32_t batch;
    uint32_t input;
    uint32_t hidden;
    uint32_t output;
};

// Sync benchmark: Finish() after each operation
double bench_sync(MeshDevice& dev, const BenchConfig& cfg) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    // Create weights and input
    auto w1 = ones(Shape({cfg.hidden, cfg.input}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto b1 = zeros(Shape({1, cfg.hidden}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto w2 = ones(Shape({cfg.output, cfg.hidden}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto b2 = zeros(Shape({1, cfg.output}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto x = ones(Shape({cfg.batch, cfg.input}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    Finish(cq);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        auto h1 = ttnn::matmul(x, w1, false, true);  Finish(cq);
        h1 = ttnn::add(h1, b1);                       Finish(cq);
        h1 = ttnn::relu(h1);                          Finish(cq);
        auto out = ttnn::matmul(h1, w2, false, true); Finish(cq);
        out = ttnn::add(out, b2);                     Finish(cq);
    }

    // Timed runs
    std::vector<double> times_ms(N_TIMED);

    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        auto h1 = ttnn::matmul(x, w1, false, true);  Finish(cq);
        h1 = ttnn::add(h1, b1);                       Finish(cq);
        h1 = ttnn::relu(h1);                          Finish(cq);
        auto out = ttnn::matmul(h1, w2, false, true); Finish(cq);
        out = ttnn::add(out, b2);                     Finish(cq);

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Return mean
    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

// Async benchmark: batch all operations, single Finish() at end
double bench_async(MeshDevice& dev, const BenchConfig& cfg) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    // Create weights and input
    auto w1 = ones(Shape({cfg.hidden, cfg.input}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto b1 = zeros(Shape({1, cfg.hidden}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto w2 = ones(Shape({cfg.output, cfg.hidden}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto b2 = zeros(Shape({1, cfg.output}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto x = ones(Shape({cfg.batch, cfg.input}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    Finish(cq);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        auto h1 = ttnn::matmul(x, w1, false, true);
        h1 = ttnn::add(h1, b1);
        h1 = ttnn::relu(h1);
        auto out = ttnn::matmul(h1, w2, false, true);
        out = ttnn::add(out, b2);
    }
    Finish(cq);

    // Timed runs
    std::vector<double> times_ms(N_TIMED);

    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        auto h1 = ttnn::matmul(x, w1, false, true);
        h1 = ttnn::add(h1, b1);
        h1 = ttnn::relu(h1);
        auto out = ttnn::matmul(h1, w2, false, true);
        out = ttnn::add(out, b2);
        Finish(cq);  // Single sync at end

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    // Return mean
    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

int main() {
    auto device = MeshDevice::create_unit_mesh(0);

    // Size configurations (all dimensions tile-aligned)
    std::vector<BenchConfig> configs = {
        {"small",  32,   128,  256,  128},
        {"medium", 256,  512,  1024, 512},
        {"large",  1024, 1024, 2048, 1024},
    };

    fmt::print("# Linear-ReLU-Linear Command Queue Benchmark\n");
    fmt::print("# Network: x[B,I] @ W1[H,I].T + b1 -> ReLU -> @ W2[O,H].T + b2\n");
    fmt::print("# Warmup: {}, Timed iterations: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");
    fmt::print("# Sync:  Finish() after each op (5 syncs per forward)\n");
    fmt::print("# Async: Batch all ops, single Finish() at end (1 sync per forward)\n");
    fmt::print("#\n");
    fmt::print("size,batch,in,hid,out,sync_ms,async_ms,speedup\n");

    for (const auto& cfg : configs) {
        double sync_ms = bench_sync(*device, cfg);
        double async_ms = bench_async(*device, cfg);
        double speedup = sync_ms / async_ms;

        fmt::print("{},{},{},{},{},{:.3f},{:.3f},{:.2f}x\n",
                   cfg.name, cfg.batch, cfg.input, cfg.hidden, cfg.output,
                   sync_ms, async_ms, speedup);
    }

    device->close();
    return 0;
}
