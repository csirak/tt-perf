// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: MLP vs FusedMLP (unfused vs fused Linear+ReLU)
// Compares training step time between unfused MLP and FusedMLP with ttnn::linear fusion.

#include "static/nn.hpp"
#include "common.hpp"

#include <chrono>
#include <vector>
#include <numeric>
#include <iomanip>

using namespace static_autograd;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;

constexpr int N_WARMUP = 10;
constexpr int N_TIMED = 50;

struct BenchConfig {
    const char* name;
    uint32_t batch;
    uint32_t in_dim;
    uint32_t hidden_dim;
    uint32_t out_dim;
};

// Benchmark unfused MLP (separate matmul + add + relu)
double bench_mlp_unfused(MeshDevice& dev, const BenchConfig& cfg) {
    // Create model
    MLP model(cfg.batch, cfg.in_dim, cfg.hidden_dim, cfg.out_dim, 0.01f, dev);

    // Create input and target
    auto x = make_full(ttnn::Shape({cfg.batch, cfg.in_dim}), 1.0f, dev);
    auto target = make_full(ttnn::Shape({cfg.batch, cfg.out_dim}), 0.5f, dev);

    // Loss buffers
    Tensor loss = make_zeros(ttnn::Shape({1, 1}), dev);
    Tensor d_loss = make_zeros(ttnn::Shape({1, 1}), dev);
    Tensor diff = make_zeros(ttnn::Shape({cfg.batch, cfg.out_dim}), dev);

    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        Graph g;
        auto* x_node = g.leaf(&x, nullptr, false);
        auto* pred = model.forward(g, x_node);
        auto* loss_node = mse(g, pred, &target, &loss, &d_loss, &diff);
        g.build_topo(loss_node);
        g.zero_grad();
        g.backward(loss_node);
        model.sgd_step(0.01f);
        model.zero_grad();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        Graph g;
        auto* x_node = g.leaf(&x, nullptr, false);
        auto* pred = model.forward(g, x_node);
        auto* loss_node = mse(g, pred, &target, &loss, &d_loss, &diff);
        g.build_topo(loss_node);
        g.zero_grad();
        g.backward(loss_node);
        model.sgd_step(0.01f);
        model.zero_grad();

        tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

// Benchmark fused MLP (ttnn::linear with activation="relu")
double bench_mlp_fused(MeshDevice& dev, const BenchConfig& cfg) {
    // Create model
    FusedMLP model(cfg.batch, cfg.in_dim, cfg.hidden_dim, cfg.out_dim, 0.01f, dev);

    // Create input and target
    auto x = make_full(ttnn::Shape({cfg.batch, cfg.in_dim}), 1.0f, dev);
    auto target = make_full(ttnn::Shape({cfg.batch, cfg.out_dim}), 0.5f, dev);

    // Loss buffers
    Tensor loss = make_zeros(ttnn::Shape({1, 1}), dev);
    Tensor d_loss = make_zeros(ttnn::Shape({1, 1}), dev);
    Tensor diff = make_zeros(ttnn::Shape({cfg.batch, cfg.out_dim}), dev);

    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        Graph g;
        auto* x_node = g.leaf(&x, nullptr, false);
        auto* pred = model.forward(g, x_node);
        auto* loss_node = mse(g, pred, &target, &loss, &d_loss, &diff);
        g.build_topo(loss_node);
        g.zero_grad();
        g.backward(loss_node);
        model.sgd_step(0.01f);
        model.zero_grad();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        Graph g;
        auto* x_node = g.leaf(&x, nullptr, false);
        auto* pred = model.forward(g, x_node);
        auto* loss_node = mse(g, pred, &target, &loss, &d_loss, &diff);
        g.build_topo(loss_node);
        g.zero_grad();
        g.backward(loss_node);
        model.sgd_step(0.01f);
        model.zero_grad();

        tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    // Size configurations (all dimensions tile-aligned)
    std::vector<BenchConfig> configs = {
        {"tiny",   32,   128,  256,  128},
        {"small",  32,   256,  512,  256},
        {"medium", 256,  512,  1024, 512},
        {"large",  512,  1024, 2048, 1024},
        {"xlarge", 1024, 1024, 2048, 1024},
    };

    fmt::print("# MLP vs FusedMLP Benchmark (Static Autograd)\n");
    fmt::print("# Network: x[B,I] -> Linear+ReLU[H] -> Linear[O]\n");
    fmt::print("# Warmup: {}, Timed iterations: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");
    fmt::print("# MLP:      Separate matmul + add + relu (3 ops)\n");
    fmt::print("# FusedMLP: ttnn::linear with activation='relu' (1 op)\n");
    fmt::print("#\n");
    fmt::print("size,batch,in,hid,out,unfused_ms,fused_ms,speedup\n");

    for (const auto& cfg : configs) {
        double unfused_ms = bench_mlp_unfused(device, cfg);
        double fused_ms = bench_mlp_fused(device, cfg);
        double speedup = unfused_ms / fused_ms;

        fmt::print("{},{},{},{},{},{:.3f},{:.3f},{:.2f}x\n",
                   cfg.name, cfg.batch, cfg.in_dim, cfg.hidden_dim, cfg.out_dim,
                   unfused_ms, fused_ms, speedup);
    }

    return 0;
}
