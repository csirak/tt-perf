// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Isolated benchmark: static_autograd with TTNN trace API
// For Tracy profiling comparison with traced+trace_api

#include "static/nn.hpp"
#include "traced/trace.hpp"
#include "common.hpp"

#include <iostream>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;

constexpr uint32_t batch = 1024;
constexpr uint32_t dim = 512;
constexpr int warmup = 5;
constexpr int iters = 10;  // Few iters for clear Tracy capture
constexpr float lr = 0.01f;

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    std::cout << "Benchmark: static+trace (for Tracy profiling)\n";
    std::cout << "Config: batch=" << batch << ", dim=" << dim << "\n\n";

    // Create model with persistent graph
    static_autograd::PersistentMLP model(batch, dim, lr, device);
    model.build();
    model.input = static_autograd::make_full(ttnn::Shape({batch, dim}), 0.1f, device);

    // Non-traced warmup (load kernels)
    for (int i = 0; i < 2; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    // Create trace and capture
    traced::TraceContext trace(&device);
    trace.run([&]() { model.train_step(); });

    // Warmup with trace
    for (int i = 0; i < warmup; ++i) {
        trace.run([&]() { model.train_step(); });
    }
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    std::cout << "Running " << iters << " traced iterations...\n";

    // Timed iterations
    for (int i = 0; i < iters; ++i) {
        trace.run([&]() { model.train_step(); });
    }
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    trace.release();

    std::cout << "Done.\n";
    return 0;
}
