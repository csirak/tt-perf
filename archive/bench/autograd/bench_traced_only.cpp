// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Isolated benchmark: traced MLP with TTNN trace API
// For Tracy profiling comparison with static+trace

#include "traced/mlp.hpp"
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

    std::cout << "Benchmark: traced+trace_api (for Tracy profiling)\n";
    std::cout << "Config: batch=" << batch << ", dim=" << dim << "\n\n";

    // Create model
    traced::TracedMLP<2> model(batch, dim, lr, device);
    auto x = traced::make_full(ttnn::Shape({batch, dim}), 0.1f, device);
    auto target = traced::make_full(ttnn::Shape({batch, dim}), 0.5f, device);

    // Non-traced warmup (load kernels)
    for (int i = 0; i < 2; ++i) {
        model.train_step(x, target);
    }
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    // Create trace and capture
    traced::TraceContext trace(&device);
    trace.run([&]() { model.train_step(x, target); });

    // Warmup with trace
    for (int i = 0; i < warmup; ++i) {
        trace.run([&]() { model.train_step(x, target); });
    }
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    std::cout << "Running " << iters << " traced iterations...\n";

    // Timed iterations
    for (int i = 0; i < iters; ++i) {
        trace.run([&]() { model.train_step(x, target); });
    }
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    trace.release();

    std::cout << "Done.\n";
    return 0;
}
