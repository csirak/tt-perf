// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Autograd Sweep Benchmark: Compare implementations across batch x dim
// Tests: c++ dynamic, c++ static (persistent), c++ static+trace

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

constexpr int warmup = 5;
constexpr int iters = 20;
constexpr float lr = 0.01f;

// ============================================================================
// Dynamic Autograd Benchmark
// ============================================================================
double bench_dynamic(uint32_t batch, uint32_t dim, MeshDevice& dev) {
    autograd::MLP model(dim, dim, dim, dev);
    auto x = autograd::make_ones(ttnn::Shape({batch, dim}), dev, false, "input");
    auto target = autograd::make_zeros(ttnn::Shape({batch, dim}), dev, false, "target");
    target->data = ttnn::full(ttnn::Shape({batch, dim}), 0.5f,
                              ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, dev);

    // Warmup
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

// ============================================================================
// Static Autograd (Persistent Graph) Benchmark
// ============================================================================
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

// ============================================================================
// Static + Trace API Benchmark
// ============================================================================
double bench_static_trace(uint32_t batch, uint32_t dim, MeshDevice& dev) {
    static_autograd::PersistentMLP model(batch, dim, lr, dev);
    model.build();
    model.input = static_autograd::make_full(ttnn::Shape({batch, dim}), 0.1f, dev);

    // Non-traced warmup (compile kernels)
    for (int i = 0; i < 2; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Capture trace
    auto trace_id = ttnn::operations::trace::begin_trace_capture(&dev, std::nullopt);
    model.train_step();
    ttnn::operations::trace::end_trace_capture(&dev, trace_id, std::nullopt);

    // Traced warmup
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

// ============================================================================
int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    std::cout << "Autograd Sweep Benchmark\n";
    std::cout << "========================\n";
    std::cout << "Warmup: " << warmup << ", Iters: " << iters << "\n\n";

    const uint32_t batches[] = {256, 512, 1024, 2048, 4096};
    const uint32_t dims[] = {256, 512, 1024, 2048};

    std::ofstream csv("bench_autograd_sweep.csv");
    csv << "batch,dim,dynamic,static,static_trace\n";

    for (auto batch : batches) {
        for (auto dim : dims) {
            std::cout << "batch=" << batch << ", dim=" << dim << " ... " << std::flush;

            double t_dyn = -1, t_static = -1, t_trace = -1;
            try {
                t_dyn = bench_dynamic(batch, dim, device);
            } catch (...) { t_dyn = -1; }

            try {
                t_static = bench_static(batch, dim, device);
            } catch (...) { t_static = -1; }

            try {
                t_trace = bench_static_trace(batch, dim, device);
            } catch (...) { t_trace = -1; }

            csv << batch << "," << dim << ",";
            if (t_dyn > 0) csv << std::fixed << std::setprecision(3) << t_dyn; else csv << "OOM";
            csv << ",";
            if (t_static > 0) csv << std::fixed << std::setprecision(3) << t_static; else csv << "OOM";
            csv << ",";
            if (t_trace > 0) csv << std::fixed << std::setprecision(3) << t_trace; else csv << "OOM";
            csv << "\n";
            csv.flush();

            std::cout << "dyn=" << (t_dyn > 0 ? std::to_string(t_dyn) : "OOM")
                      << " static=" << (t_static > 0 ? std::to_string(t_static) : "OOM")
                      << " trace=" << (t_trace > 0 ? std::to_string(t_trace) : "OOM") << "\n";

            tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        }
    }

    csv.close();
    std::cout << "\nResults written to bench_autograd_sweep.csv\n";
    return 0;
}
