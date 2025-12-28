// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: Linear+ReLU Forward Only (unfused vs fused)
// Isolates the forward pass to measure fusion benefit directly.

#include "static/value.hpp"
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

constexpr int N_WARMUP = 20;
constexpr int N_TIMED = 100;

struct BenchConfig {
    const char* name;
    uint32_t batch;
    uint32_t in_dim;
    uint32_t out_dim;
};

// Unfused: matmul + add + relu (3 separate ops)
double bench_unfused(MeshDevice& dev, const BenchConfig& cfg) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    auto x = ones(Shape({cfg.batch, cfg.in_dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto w = ones(Shape({cfg.out_dim, cfg.in_dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto b = zeros(Shape({1, cfg.out_dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    Finish(cq);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        auto mm = ttnn::matmul(x, w, false, true);  // x @ w.T
        auto added = ttnn::add(mm, b);               // + bias
        auto out = ttnn::relu(added);                // relu
    }
    Finish(cq);

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        auto mm = ttnn::matmul(x, w, false, true);
        auto added = ttnn::add(mm, b);
        auto out = ttnn::relu(added);
        Finish(cq);

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

// Fused: ttnn::linear with activation='relu' (1 op)
double bench_fused(MeshDevice& dev, const BenchConfig& cfg) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    auto x = ones(Shape({cfg.batch, cfg.in_dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto w = ones(Shape({cfg.out_dim, cfg.in_dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto b = zeros(Shape({1, cfg.out_dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    Finish(cq);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        auto out = ttnn::linear(x, w, b, false, true,
                                std::nullopt, std::nullopt, std::nullopt, "relu");
    }
    Finish(cq);

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        auto out = ttnn::linear(x, w, b, false, true,
                                std::nullopt, std::nullopt, std::nullopt, "relu");
        Finish(cq);

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

int main() {
    auto device = MeshDevice::create_unit_mesh(0);

    // Size configurations
    std::vector<BenchConfig> configs = {
        {"tiny",   32,   128,  256},
        {"small",  32,   256,  512},
        {"medium", 256,  512,  1024},
        {"large",  512,  1024, 2048},
        {"xlarge", 1024, 1024, 2048},
    };

    fmt::print("# Linear+ReLU Forward Benchmark (Unfused vs Fused)\n");
    fmt::print("# Network: x[B,I] @ W[O,I].T + b -> ReLU\n");
    fmt::print("# Warmup: {}, Timed iterations: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");
    fmt::print("# Unfused: matmul + add + relu (3 ops)\n");
    fmt::print("# Fused:   ttnn::linear(..., activation='relu') (1 op)\n");
    fmt::print("#\n");
    fmt::print("size,batch,in,out,unfused_ms,fused_ms,speedup\n");

    for (const auto& cfg : configs) {
        double unfused_ms = bench_unfused(*device, cfg);
        double fused_ms = bench_fused(*device, cfg);
        double speedup = unfused_ms / fused_ms;

        fmt::print("{},{},{},{},{:.3f},{:.3f},{:.2f}x\n",
                   cfg.name, cfg.batch, cfg.in_dim, cfg.out_dim,
                   unfused_ms, fused_ms, speedup);
    }

    device->close();
    return 0;
}
