// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Scaling Sweep Benchmark: Batch x Dimension
// Compares our traced autograd across different scales.
// Outputs CSV for comparison with TTML.

#include "traced/mlp.hpp"
#include "traced/trace.hpp"
#include "common.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;

// Run benchmark for 2-layer MLP with given batch and dim
double benchmark_config(uint32_t batch, uint32_t dim, MeshDevice& dev, int warmup, int iters) {
    // Create input and target
    auto x = traced::make_full(ttnn::Shape({batch, dim}), 1.0f, dev);
    auto target = traced::make_full(ttnn::Shape({batch, dim}), 0.5f, dev);

    // Create 2-layer MLP (in=hidden=out=dim for simplicity)
    traced::TracedMLP<2> model(batch, dim, 0.01f, dev);

    // NON-TRACED warmup first - ensures kernel binaries are loaded
    for (int i = 0; i < 2; ++i) {
        model.train_step(x, target);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Create trace context and capture
    traced::TraceContext trace(&dev);
    trace.run([&]() { model.train_step(x, target); });

    // Traced warmup
    for (int i = 0; i < warmup; ++i) {
        trace.run([&]() { model.train_step(x, target); });
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Measure traced execution
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        trace.run([&]() { model.train_step(x, target); });
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    trace.release();
    return ms;
}

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    constexpr int warmup = 5;
    constexpr int iters = 20;

    // Sweep parameters
    const uint32_t batches[] = {256, 512, 1024, 2048, 4096};
    const uint32_t dims[] = {256, 512, 1024, 2048};

    std::cout << "Scaling Sweep Benchmark (Our Traced Autograd)\n";
    std::cout << "==============================================\n";
    std::cout << "Warmup: " << warmup << ", Iters: " << iters << "\n\n";

    // Open CSV file
    std::ofstream csv("bench_sweep_ours.csv");
    csv << "batch,dim,ms_per_iter\n";

    for (auto batch : batches) {
        for (auto dim : dims) {
            std::cout << "batch=" << batch << ", dim=" << dim << " ... " << std::flush;

            try {
                double ms = benchmark_config(batch, dim, device, warmup, iters);
                csv << batch << "," << dim << "," << std::fixed << std::setprecision(3) << ms << "\n";
                csv.flush();
                std::cout << ms << " ms/iter\n";
            } catch (const std::exception& e) {
                csv << batch << "," << dim << ",OOM\n";
                csv.flush();
                std::cout << "OOM or error: " << e.what() << "\n";
                try {
                    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
                } catch (...) {}
            }
        }
    }

    csv.close();
    std::cout << "\nResults written to bench_sweep_ours.csv\n";

    return 0;
}
