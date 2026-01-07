// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: Compare dynamic, static_autograd, and traced autograd implementations
// Measures training step time (forward + backward + SGD) for 2-layer MLP

#include "dynamic/autograd.hpp"
#include "dynamic/ops.hpp"
#include "dynamic/nn.hpp"
#include "static/nn.hpp"
#include "traced/mlp.hpp"
#include "traced/trace.hpp"
#include "common.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;

// Benchmark configuration
constexpr uint32_t batch = 1024;
constexpr uint32_t dim = 512;
constexpr int warmup = 20;
constexpr int iters = 200;
constexpr float lr = 0.01f;

// ============================================================================
// Dynamic Autograd Benchmark
// ============================================================================
double bench_dynamic(MeshDevice& dev) {
    // Create model
    autograd::MLP model(dim, dim, dim, dev);

    // Create input and target
    auto x = autograd::make_ones(ttnn::Shape({batch, dim}), dev, false, "input");
    auto target = autograd::make_zeros(ttnn::Shape({batch, dim}), dev, false, "target");
    target->data = ttnn::full(ttnn::Shape({batch, dim}), 0.5f,
                              ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, dev);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        auto pred = model.forward(x);
        auto loss = autograd::mse_loss(pred, target);
        autograd::backward(loss, dev);

        // SGD update
        for (auto& p : model.parameters()) {
            if (p->grad.has_value()) {
                p->data = ttnn::subtract(p->data, ttnn::multiply(p->grad.value(), lr));
                p->grad = std::nullopt;
            }
        }
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Timed iterations
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

// ============================================================================
// Static Autograd Benchmark
// ============================================================================
double bench_static(MeshDevice& dev) {
    // Create model with pre-allocated buffers
    static_autograd::MLP model(batch, dim, dim, dim, 0.01f, dev);

    // Create input and target buffers
    auto input = static_autograd::make_full(ttnn::Shape({batch, dim}), 0.1f, dev);
    auto target = static_autograd::make_full(ttnn::Shape({batch, dim}), 0.5f, dev);
    auto d_input = static_autograd::make_zeros(ttnn::Shape({batch, dim}), dev);

    // Loss buffers
    auto loss = static_autograd::make_zeros(ttnn::Shape({1, 1}), dev);
    auto d_loss = static_autograd::make_zeros(ttnn::Shape({1, 1}), dev);
    auto diff = static_autograd::make_zeros(ttnn::Shape({batch, dim}), dev);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        static_autograd::Graph g;
        auto* x = g.leaf(&input, &d_input, false);
        auto* pred = model.forward(g, x);
        auto* loss_v = static_autograd::mse(g, pred, &target, &loss, &d_loss, &diff);
        g.zero_grad();
        model.zero_grad();
        g.backward(loss_v);
        model.sgd_step(lr);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Timed iterations
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        static_autograd::Graph g;
        auto* x = g.leaf(&input, &d_input, false);
        auto* pred = model.forward(g, x);
        auto* loss_v = static_autograd::mse(g, pred, &target, &loss, &d_loss, &diff);
        g.zero_grad();
        model.zero_grad();
        g.backward(loss_v);
        model.sgd_step(lr);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

// ============================================================================
// Static Autograd Benchmark (Persistent Graph)
// ============================================================================
double bench_static_persistent(MeshDevice& dev) {
    // Create model with persistent graph
    static_autograd::PersistentMLP model(batch, dim, lr, dev);

    // Build graph once
    model.build();

    // Set input data
    model.input = static_autograd::make_full(ttnn::Shape({batch, dim}), 0.1f, dev);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Timed iterations
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

// ============================================================================
// Static Autograd Benchmark (Persistent Graph + TTNN Trace API)
// ============================================================================
double bench_static_with_trace(MeshDevice& dev) {
    // Create model with persistent graph
    static_autograd::PersistentMLP model(batch, dim, lr, dev);

    // Build graph once
    model.build();

    // Set input data
    model.input = static_autograd::make_full(ttnn::Shape({batch, dim}), 0.1f, dev);

    // Non-traced warmup (required to load kernels before trace capture)
    for (int i = 0; i < 2; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Create trace context and capture
    traced::TraceContext trace(&dev);
    trace.run([&]() { model.train_step(); });

    // Warmup with trace replay
    for (int i = 0; i < warmup; ++i) {
        trace.run([&]() { model.train_step(); });
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Timed iterations with trace replay
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        trace.run([&]() { model.train_step(); });
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    trace.release();

    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

// ============================================================================
// Traced Benchmark (no trace API)
// ============================================================================
double bench_traced(MeshDevice& dev) {
    // Create model with pre-allocated buffers
    traced::TracedMLP<2> model(batch, dim, lr, dev);

    // Create input and target
    auto x = traced::make_full(ttnn::Shape({batch, dim}), 0.1f, dev);
    auto target = traced::make_full(ttnn::Shape({batch, dim}), 0.5f, dev);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        model.train_step(x, target);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Timed iterations
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        model.train_step(x, target);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

// ============================================================================
// Traced Benchmark (with TTNN trace API)
// ============================================================================
double bench_traced_with_trace(MeshDevice& dev) {
    // Create model with pre-allocated buffers
    traced::TracedMLP<2> model(batch, dim, lr, dev);

    // Create input and target
    auto x = traced::make_full(ttnn::Shape({batch, dim}), 0.1f, dev);
    auto target = traced::make_full(ttnn::Shape({batch, dim}), 0.5f, dev);

    // Non-traced warmup (required to load kernels before trace capture)
    for (int i = 0; i < 2; ++i) {
        model.train_step(x, target);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Create trace context and capture
    traced::TraceContext trace(&dev);
    trace.run([&]() { model.train_step(x, target); });

    // Warmup with trace replay
    for (int i = 0; i < warmup; ++i) {
        trace.run([&]() { model.train_step(x, target); });
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Timed iterations with trace replay
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        trace.run([&]() { model.train_step(x, target); });
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    trace.release();

    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    std::cout << "Autograd Benchmark: 2-layer MLP\n";
    std::cout << "================================\n";
    std::cout << "Config: batch=" << batch << ", dim=" << dim
              << ", warmup=" << warmup << ", iters=" << iters << "\n\n";

    std::cout << "Running benchmarks...\n\n";

    // Run benchmarks
    std::cout << "  dynamic ... " << std::flush;
    double t_dynamic = bench_dynamic(device);
    std::cout << t_dynamic << " ms/iter\n";

    std::cout << "  static (new graph) ... " << std::flush;
    double t_static = bench_static(device);
    std::cout << t_static << " ms/iter\n";

    std::cout << "  static (persistent) ... " << std::flush;
    double t_static_persist = bench_static_persistent(device);
    std::cout << t_static_persist << " ms/iter\n";

    std::cout << "  static+trace ... " << std::flush;
    double t_static_trace = bench_static_with_trace(device);
    std::cout << t_static_trace << " ms/iter\n";

    std::cout << "  traced ... " << std::flush;
    double t_traced = bench_traced(device);
    std::cout << t_traced << " ms/iter\n";

    std::cout << "  traced+trace_api ... " << std::flush;
    double t_traced_api = bench_traced_with_trace(device);
    std::cout << t_traced_api << " ms/iter\n";

    // Print results table
    std::cout << "\n================================\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::left << std::setw(22) << "Method"
              << std::right << std::setw(12) << "ms/iter"
              << std::setw(12) << "relative" << "\n";
    std::cout << std::string(46, '-') << "\n";

    double baseline = t_dynamic;
    std::cout << std::left << std::setw(22) << "dynamic"
              << std::right << std::setw(12) << t_dynamic
              << std::setw(10) << "1.00" << "x (baseline)\n";
    std::cout << std::left << std::setw(22) << "static (new graph)"
              << std::right << std::setw(12) << t_static
              << std::setw(10) << (baseline / t_static) << "x\n";
    std::cout << std::left << std::setw(22) << "static (persistent)"
              << std::right << std::setw(12) << t_static_persist
              << std::setw(10) << (baseline / t_static_persist) << "x\n";
    std::cout << std::left << std::setw(22) << "static+trace"
              << std::right << std::setw(12) << t_static_trace
              << std::setw(10) << (baseline / t_static_trace) << "x\n";
    std::cout << std::left << std::setw(22) << "traced"
              << std::right << std::setw(12) << t_traced
              << std::setw(10) << (baseline / t_traced) << "x\n";
    std::cout << std::left << std::setw(22) << "traced+trace_api"
              << std::right << std::setw(12) << t_traced_api
              << std::setw(10) << (baseline / t_traced_api) << "x\n";

    // Write CSV
    std::ofstream csv("bench_autograd_results.csv");
    csv << "method,ms_per_iter\n";
    csv << "dynamic," << t_dynamic << "\n";
    csv << "static_new_graph," << t_static << "\n";
    csv << "static_persistent," << t_static_persist << "\n";
    csv << "static_trace," << t_static_trace << "\n";
    csv << "traced," << t_traced << "\n";
    csv << "traced_trace_api," << t_traced_api << "\n";
    csv.close();

    std::cout << "\nResults written to bench_autograd_results.csv\n";

    return 0;
}
