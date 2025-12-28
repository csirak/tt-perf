// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: MLP with restricted core grids
// Tests if we can run computations on a subset of cores for data parallel workloads.
// Measures throughput and efficiency per core for different grid sizes.

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

constexpr int N_WARMUP = 10;
constexpr int N_TIMED = 50;

struct GridConfig {
    const char* name;
    uint32_t x;  // columns
    uint32_t y;  // rows
};

struct SizeConfig {
    const char* name;
    uint32_t batch;
    uint32_t dim;
};

// 2-layer MLP forward + backward with specific core grid
// Forward:  h = relu(x @ W1.T + b1), out = h @ W2.T + b2
// Backward: dW2, dW1, dx
double bench_mlp_grid(MeshDevice& dev, const SizeConfig& size,
                      const std::optional<CoreGrid>& core_grid) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    // Create tensors
    auto x = ones(Shape({size.batch, size.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto w1 = ones(Shape({size.dim, size.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto b1 = zeros(Shape({1, size.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto w2 = ones(Shape({size.dim, size.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto b2 = zeros(Shape({1, size.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto dout = ones(Shape({size.batch, size.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    Finish(cq);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        // Forward
        auto mm1 = ttnn::matmul(x, w1, false, true, std::nullopt, std::nullopt,
                                std::nullopt, std::nullopt, std::nullopt, core_grid);
        auto h_pre = ttnn::add(mm1, b1);
        auto h = ttnn::relu(h_pre);
        auto mm2 = ttnn::matmul(h, w2, false, true, std::nullopt, std::nullopt,
                                std::nullopt, std::nullopt, std::nullopt, core_grid);
        auto out = ttnn::add(mm2, b2);

        // Backward
        auto dh = ttnn::matmul(dout, w2, false, false, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt, core_grid);
        auto relu_mask = ttnn::gtz(h_pre);
        auto dh_pre = ttnn::multiply(dh, relu_mask);
        auto dx = ttnn::matmul(dh_pre, w1, false, false, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt, core_grid);
        // Gradient updates (dW = dout.T @ x, etc.) - simplified, skip for timing
    }
    Finish(cq);

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        // Forward
        auto mm1 = ttnn::matmul(x, w1, false, true, std::nullopt, std::nullopt,
                                std::nullopt, std::nullopt, std::nullopt, core_grid);
        auto h_pre = ttnn::add(mm1, b1);
        auto h = ttnn::relu(h_pre);
        auto mm2 = ttnn::matmul(h, w2, false, true, std::nullopt, std::nullopt,
                                std::nullopt, std::nullopt, std::nullopt, core_grid);
        auto out = ttnn::add(mm2, b2);

        // Backward
        auto dh = ttnn::matmul(dout, w2, false, false, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt, core_grid);
        auto relu_mask = ttnn::gtz(h_pre);
        auto dh_pre = ttnn::multiply(dh, relu_mask);
        auto dx = ttnn::matmul(dh_pre, w1, false, false, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt, core_grid);

        Finish(cq);

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

int main() {
    auto device = MeshDevice::create_unit_mesh(0);

    // Get device core grid size
    auto grid_size = device->compute_with_storage_grid_size();
    fmt::print("# Device core grid: {}x{} = {} cores\n", grid_size.x, grid_size.y,
               grid_size.x * grid_size.y);

    // Grid configurations to test
    std::vector<GridConfig> grids = {
        {"full", 0, 0},      // std::nullopt - use full grid
        {"8x8",  8, 8},      // 64 cores (may fail if device has fewer)
        {"4x8",  4, 8},      // 32 cores
        {"4x4",  4, 4},      // 16 cores
        {"2x4",  2, 4},      // 8 cores
        {"2x2",  2, 2},      // 4 cores
    };

    // Size configurations
    std::vector<SizeConfig> sizes = {
        {"small",  32,   256},   // Small batch, small dim
        {"medium", 256,  512},   // Medium
        {"large",  1024, 1024},  // Large batch and dim
    };

    fmt::print("# MLP Core Grid Benchmark\n");
    fmt::print("# Network: x[B,D] -> Linear+ReLU[D] -> Linear[D]\n");
    fmt::print("# Measures forward + backward pass\n");
    fmt::print("# Warmup: {}, Timed iterations: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");
    fmt::print("size,batch,dim,grid,cores,ms,throughput,efficiency\n");

    for (const auto& size : sizes) {
        for (const auto& grid : grids) {
            std::optional<CoreGrid> core_grid;
            uint32_t cores;

            if (grid.x == 0 && grid.y == 0) {
                // Use full grid
                core_grid = std::nullopt;
                cores = grid_size.x * grid_size.y;
            } else {
                // Skip if grid is larger than device
                if (grid.x > grid_size.x || grid.y > grid_size.y) {
                    fmt::print("{},{},{},{},{},SKIP,SKIP,SKIP\n",
                               size.name, size.batch, size.dim, grid.name,
                               grid.x * grid.y);
                    continue;
                }
                core_grid = CoreGrid(grid.x, grid.y);
                cores = grid.x * grid.y;
            }

            double ms = bench_mlp_grid(*device, size, core_grid);
            double throughput = size.batch / ms * 1000.0;  // samples/sec
            double efficiency = throughput / cores;         // samples/sec/core

            fmt::print("{},{},{},{},{},{:.3f},{:.1f},{:.2f}\n",
                       size.name, size.batch, size.dim, grid.name, cores,
                       ms, throughput, efficiency);
        }
        fmt::print("#\n");  // Separator between sizes
    }

    device->close();
    return 0;
}
