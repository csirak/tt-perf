// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Single Shape Benchmark: Test one batch×dim configuration
// Usage: Set BENCH_BATCH and BENCH_DIM env vars or use defaults

#include "dynamic/autograd.hpp"
#include "dynamic/ops.hpp"
#include "dynamic/nn.hpp"
#include "static/nn.hpp"
#include "traced/mlp.hpp"
#include "traced/trace.hpp"
#include "common.hpp"

#include <iostream>
#include <iomanip>
#include <cstdlib>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;

constexpr int warmup = 5;
constexpr int iters = 20;
constexpr float lr = 0.01f;

double bench_dynamic(uint32_t batch, uint32_t dim, MeshDevice& dev) {
    autograd::MLP model(dim, dim, dim, dev);
    auto x = autograd::make_ones(ttnn::Shape({batch, dim}), dev, false, "input");
    auto target = autograd::make_zeros(ttnn::Shape({batch, dim}), dev, false, "target");
    target->data = ttnn::full(ttnn::Shape({batch, dim}), 0.5f,
                              ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, dev);

    for (int i = 0; i < warmup; ++i) {
        auto pred = model.forward(x);
        auto loss = autograd::mse_loss(pred, target);
        autograd::backward(loss, dev);
        for (auto& p : model.parameters()) {
            if (p->grad.has_value()) {
                p->data = ttnn::subtract(p->data, ttnn::multiply(p->grad.value(), lr));
                p->grad = std::nullopt;
            }
        }
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        auto pred = model.forward(x);
        auto loss = autograd::mse_loss(pred, target);
        autograd::backward(loss, dev);
        for (auto& p : model.parameters()) {
            if (p->grad.has_value()) {
                p->data = ttnn::subtract(p->data, ttnn::multiply(p->grad.value(), lr));
                p->grad = std::nullopt;
            }
        }
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

double bench_static(uint32_t batch, uint32_t dim, MeshDevice& dev) {
    static_autograd::PersistentMLP model(batch, dim, lr, dev);
    model.build();
    model.input = static_autograd::make_full(ttnn::Shape({batch, dim}), 0.1f, dev);

    for (int i = 0; i < warmup; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

double bench_static_trace(uint32_t batch, uint32_t dim, MeshDevice& dev) {
    static_autograd::PersistentMLP model(batch, dim, lr, dev);
    model.build();
    model.input = static_autograd::make_full(ttnn::Shape({batch, dim}), 0.1f, dev);

    for (int i = 0; i < 2; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    auto trace_id = ttnn::operations::trace::begin_trace_capture(&dev, std::nullopt);
    model.train_step();
    ttnn::operations::trace::end_trace_capture(&dev, trace_id, std::nullopt);

    for (int i = 0; i < warmup; ++i) {
        ttnn::operations::trace::execute_trace(&dev, trace_id, std::nullopt, false);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        ttnn::operations::trace::execute_trace(&dev, trace_id, std::nullopt, false);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    ttnn::operations::trace::release_trace(&dev, trace_id);
    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

int main() {
    // Get batch and dim from environment or use defaults
    uint32_t batch = 2048;
    uint32_t dim = 2048;

    if (const char* env_batch = std::getenv("BENCH_BATCH")) {
        batch = std::stoul(env_batch);
    }
    if (const char* env_dim = std::getenv("BENCH_DIM")) {
        dim = std::stoul(env_dim);
    }

    std::cout << "Single Shape Benchmark: batch=" << batch << ", dim=" << dim << "\n";
    std::cout << "========================================\n";

    test::DeviceGuard dg;
    auto& device = dg.get();

    std::cout << "  dynamic ... " << std::flush;
    try {
        double t = bench_dynamic(batch, dim, device);
        std::cout << std::fixed << std::setprecision(3) << t << " ms\n";
    } catch (...) {
        std::cout << "OOM\n";
    }

    std::cout << "  static ... " << std::flush;
    try {
        double t = bench_static(batch, dim, device);
        std::cout << std::fixed << std::setprecision(3) << t << " ms\n";
    } catch (...) {
        std::cout << "OOM\n";
    }

    std::cout << "  static+trace ... " << std::flush;
    try {
        double t = bench_static_trace(batch, dim, device);
        std::cout << std::fixed << std::setprecision(3) << t << " ms\n";
    } catch (...) {
        std::cout << "OOM\n";
    }

    return 0;
}
