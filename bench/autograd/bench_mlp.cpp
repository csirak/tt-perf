// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MLP Benchmark: Layers x Dimensions
// Measures traced training step time across different configurations.
// Outputs CSV for analysis and visualization.

#include "traced/mlp.hpp"
#include "traced/trace.hpp"
#include "common.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <vector>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;

template<size_t N>
double benchmark_mlp(uint32_t batch, uint32_t dim, MeshDevice& dev, int warmup, int iters) {
    // Create input and target BEFORE trace (these are static)
    auto x = traced::make_full(ttnn::Shape({batch, dim}), 1.0f, dev);
    auto target = traced::make_full(ttnn::Shape({batch, dim}), 0.5f, dev);

    // Create model (allocates all buffers)
    traced::TracedMLP<N> model(batch, dim, 0.01f, dev);

    // Create trace context - will be automatically released when it goes out of scope
    traced::TraceContext trace(&dev);

    // Capture trace on first run
    trace.run([&]() { model.train_step(x, target); });

    // Warmup iterations (replay trace)
    for (int i = 0; i < warmup; ++i) {
        trace.run([&]() { model.train_step(x, target); });
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Timed iterations
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        trace.run([&]() { model.train_step(x, target); });
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    double ms_per_iter = std::chrono::duration<double, std::milli>(end - start).count() / iters;

    // Explicitly release trace before returning (destructor will also do this)
    trace.release();

    return ms_per_iter;
}

// Helper to run benchmarks for a specific layer count
template<size_t N>
void run_layer_benchmarks(uint32_t batch, const std::vector<uint32_t>& dims,
                          MeshDevice& dev, int warmup, int iters,
                          std::ostream& csv) {
    for (uint32_t dim : dims) {
        // Ensure clean state before each benchmark
        tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

        std::cout << "  layers=" << N << ", dim=" << dim << " ... " << std::flush;
        try {
            double ms = benchmark_mlp<N>(batch, dim, dev, warmup, iters);
            csv << N << "," << dim << "," << std::fixed << std::setprecision(3) << ms << "\n";
            csv.flush();  // Ensure results are written
            std::cout << ms << " ms/iter\n";
        } catch (const std::exception& e) {
            csv << N << "," << dim << ",OOM\n";
            csv.flush();
            std::cout << "OOM or error\n";
            // Sync to clear any pending operations
            try {
                tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
            } catch (...) {}
        }
    }
}

// Run a single benchmark configuration
template<size_t N>
void run_single_benchmark(uint32_t batch, uint32_t dim, MeshDevice& dev,
                          int warmup, int iters, std::ostream& csv) {
    std::cout << "  layers=" << N << ", dim=" << dim << " ... " << std::flush;

    // Create input and target
    auto x = traced::make_full(ttnn::Shape({batch, dim}), 1.0f, dev);
    auto target = traced::make_full(ttnn::Shape({batch, dim}), 0.5f, dev);

    // Create model
    traced::TracedMLP<N> model(batch, dim, 0.01f, dev);

    // NON-TRACED warmup first - ensures kernel binaries are loaded
    // This is required before trace capture!
    for (int i = 0; i < 2; ++i) {
        model.train_step(x, target);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Now create trace context and capture
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

    csv << N << "," << dim << "," << std::fixed << std::setprecision(3) << ms << "\n";
    csv.flush();
    std::cout << ms << " ms/iter\n";

    // Release trace explicitly
    trace.release();
}

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    // Benchmark configuration
    constexpr int warmup = 5;
    constexpr int iters = 20;
    constexpr uint32_t batch = 1024;

    std::cout << "MLP Benchmark: Layers x Dimensions\n";
    std::cout << "===================================\n";
    std::cout << "Batch: " << batch << ", Warmup: " << warmup << ", Iters: " << iters << "\n\n";

    // Open CSV file
    std::ofstream csv("bench_mlp_results.csv");
    csv << "layers,dim,ms_per_iter\n";

    // Run benchmarks one at a time (must be explicit for template instantiation)
    // Start with smaller configs first
    std::cout << "Running benchmarks...\n";

    run_single_benchmark<2>(batch, 256, device, warmup, iters, csv);
    run_single_benchmark<2>(batch, 512, device, warmup, iters, csv);
    run_single_benchmark<2>(batch, 1024, device, warmup, iters, csv);
    run_single_benchmark<2>(batch, 2048, device, warmup, iters, csv);

    run_single_benchmark<4>(batch, 256, device, warmup, iters, csv);
    run_single_benchmark<4>(batch, 512, device, warmup, iters, csv);
    run_single_benchmark<4>(batch, 1024, device, warmup, iters, csv);
    run_single_benchmark<4>(batch, 2048, device, warmup, iters, csv);

    run_single_benchmark<8>(batch, 256, device, warmup, iters, csv);
    run_single_benchmark<8>(batch, 512, device, warmup, iters, csv);
    run_single_benchmark<8>(batch, 1024, device, warmup, iters, csv);
    run_single_benchmark<8>(batch, 2048, device, warmup, iters, csv);

    run_single_benchmark<16>(batch, 256, device, warmup, iters, csv);
    run_single_benchmark<16>(batch, 512, device, warmup, iters, csv);
    run_single_benchmark<16>(batch, 1024, device, warmup, iters, csv);
    run_single_benchmark<16>(batch, 2048, device, warmup, iters, csv);

    csv.close();

    std::cout << "\n===================================\n";
    std::cout << "Results written to bench_mlp_results.csv\n";

    return 0;
}
