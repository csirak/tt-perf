// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Shape Sweep Benchmark: Find optimal shapes for maximum peak utilization
// Tests various (batch, dim) combinations to find highest TFLOPs

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <tt-metalium/distributed.hpp>

#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace ttnn;
using namespace tt::tt_metal::distributed;

constexpr int N_WARMUP = 5;
constexpr int N_TIMED = 10;
constexpr double PEAK_TFLOPS = 52.0;  // Measured peak from STATUS.md

struct ShapeConfig {
    uint32_t batch;
    uint32_t dim;
    const char* name;
};

struct Result {
    uint32_t batch;
    uint32_t dim;
    double flops_G;
    double time_ms;
    double tflops;
    double pct_peak;
};

int main() {
    auto device = MeshDevice::create_unit_mesh(0);
    MeshCommandQueue& cq = device->mesh_command_queue();

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t total_cores = grid_size.x * grid_size.y;

    fmt::print("# Shape Sweep Benchmark\n");
    fmt::print("# Device grid: {}x{} = {} cores\n", grid_size.x, grid_size.y, total_cores);
    fmt::print("# Peak reference: {:.1f} TFLOPs\n", PEAK_TFLOPS);
    fmt::print("# Warmup: {}, Timed: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");

    // Shapes to test - all tile-aligned (multiples of 32)
    // Avoiding 4096 due to performance cliff
    std::vector<ShapeConfig> shapes = {
        // Small (dispatch overhead dominates)
        {256, 512, "small"},
        {512, 512, "small"},

        // Medium
        {512, 1024, "medium"},
        {1024, 1024, "medium"},

        // Large
        {1024, 2048, "large"},
        {2048, 2048, "large"},

        // Very large (approaching peak)
        {2048, 4032, "xlarge"},
        {4032, 4032, "xlarge"},

        // Square configurations
        {1024, 1024, "square"},
        {2048, 2048, "square"},
        {3072, 3072, "square"},
    };

    std::vector<Result> results;

    for (const auto& cfg : shapes) {
        fmt::print("Testing batch={}, dim={}...\n", cfg.batch, cfg.dim);

        // Create tensors
        auto x = ones(Shape({cfg.batch, cfg.dim}), DataType::BFLOAT16, TILE_LAYOUT, *device);
        auto w = ones(Shape({cfg.dim, cfg.dim}), DataType::BFLOAT16, TILE_LAYOUT, *device);
        Finish(cq);

        // Warmup
        for (int i = 0; i < N_WARMUP; i++) {
            auto out = ttnn::matmul(x, w);
            Finish(cq);
        }

        // Timed runs
        std::vector<double> times(N_TIMED);
        for (int t = 0; t < N_TIMED; t++) {
            auto start = std::chrono::high_resolution_clock::now();
            auto out = ttnn::matmul(x, w);
            Finish(cq);
            auto end = std::chrono::high_resolution_clock::now();
            times[t] = std::chrono::duration<double, std::milli>(end - start).count();
        }

        // Use median for stability
        std::sort(times.begin(), times.end());
        double median_ms = times[N_TIMED / 2];

        // FLOPs = 2 * M * K * N for matmul (multiply-accumulate)
        double flops_G = 2.0 * cfg.batch * cfg.dim * cfg.dim / 1e9;
        double tflops = flops_G / median_ms;
        double pct_peak = 100.0 * tflops / PEAK_TFLOPS;

        results.push_back({cfg.batch, cfg.dim, flops_G, median_ms, tflops, pct_peak});

        // Cleanup tensors to avoid OOM on large shapes
        x.deallocate();
        w.deallocate();
        Finish(cq);
    }

    // Print results
    fmt::print("\n## Results\n");
    fmt::print("batch,dim,flops_G,time_ms,tflops,pct_peak\n");

    for (const auto& r : results) {
        fmt::print("{},{},{:.2f},{:.3f},{:.2f},{:.1f}%\n",
            r.batch, r.dim, r.flops_G, r.time_ms, r.tflops, r.pct_peak);
    }

    // Find best configuration
    auto best = std::max_element(results.begin(), results.end(),
        [](const Result& a, const Result& b) { return a.pct_peak < b.pct_peak; });

    fmt::print("\n## Best Configuration\n");
    fmt::print("batch={}, dim={}: {:.2f} TFLOPs ({:.1f}% of peak)\n",
        best->batch, best->dim, best->tflops, best->pct_peak);

    // Print efficiency analysis
    fmt::print("\n## Efficiency Analysis\n");
    fmt::print("# Shapes sorted by % of peak:\n");

    std::vector<Result> sorted = results;
    std::sort(sorted.begin(), sorted.end(),
        [](const Result& a, const Result& b) { return a.pct_peak > b.pct_peak; });

    for (const auto& r : sorted) {
        fmt::print("#   {}x{}: {:.1f}% ({:.2f} TFLOPs, {:.3f}ms)\n",
            r.batch, r.dim, r.pct_peak, r.tflops, r.time_ms);
    }

    device->close();
    return 0;
}
