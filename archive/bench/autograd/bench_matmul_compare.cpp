// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Matmul Comparison: Full Grid vs 7x Partitioned
// Compares throughput of one large matmul vs 7 parallel smaller matmuls
//
// This tests whether partitioning gives better throughput for data-parallel workloads

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>

#include <chrono>
#include <vector>
#include <numeric>
#include <array>

using namespace ttnn;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

constexpr int N_WARMUP = 3;
constexpr int N_TIMED = 5;
constexpr int N_PARALLEL = 7;

int main() {
    auto device = MeshDevice::create_unit_mesh(0);
    MeshCommandQueue& cq = device->mesh_command_queue();

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t total_cores = grid_size.x * grid_size.y;

    fmt::print("# Matmul Comparison: Full Grid vs 7x Partitioned\n");
    fmt::print("# Device grid: {}x{} = {} cores\n", grid_size.x, grid_size.y, total_cores);
    fmt::print("# Warmup: {}, Timed: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");

    // ============================================================
    // Test 1: Full grid - one large matmul (batch=1792, same total work)
    // ============================================================
    uint32_t batch_full = 256 * N_PARALLEL;  // 1792
    uint32_t dim = 512;

    auto x_full = ones(Shape({batch_full, dim}), DataType::BFLOAT16, TILE_LAYOUT, *device);
    auto w_full = ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, *device);
    Finish(cq);

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        auto out = ttnn::matmul(x_full, w_full);
        Finish(cq);
    }

    // Timed
    std::vector<double> times_full(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto out = ttnn::matmul(x_full, w_full);
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_full[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_full = std::accumulate(times_full.begin(), times_full.end(), 0.0) / N_TIMED;
    double throughput_full = batch_full / avg_full * 1000.0;

    fmt::print("## Full Grid (1 large matmul, {} batch)\n", batch_full);
    fmt::print("avg_ms,throughput_samples_sec\n");
    fmt::print("{:.3f},{:.1f}\n", avg_full, throughput_full);
    fmt::print("#\n");

    // ============================================================
    // Test 2: 7x Partitioned - seven smaller matmuls in parallel
    // ============================================================
    uint32_t batch_part = 256;

    // Create 7 input tensors
    std::array<Tensor, N_PARALLEL> x_parts;
    std::array<Tensor, N_PARALLEL> w_parts;
    for (int i = 0; i < N_PARALLEL; i++) {
        x_parts[i] = ones(Shape({batch_part, dim}), DataType::BFLOAT16, TILE_LAYOUT, *device);
        w_parts[i] = ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, *device);
    }
    Finish(cq);

    // All grids uniform 2×4
    std::array<CoreGrid, N_PARALLEL> grids = {{
        CoreGrid(2, 4), CoreGrid(2, 4), CoreGrid(2, 4), CoreGrid(2, 4),
        CoreGrid(2, 4), CoreGrid(2, 4), CoreGrid(2, 4)
    }};

    // Non-overlapping regions (same as bench_7x2x4_uniform)
    std::array<CoreRangeSet, N_PARALLEL> core_ranges = {{
        CoreRangeSet({CoreRange({0, 0}, {1, 3})}),
        CoreRangeSet({CoreRange({2, 0}, {3, 3})}),
        CoreRangeSet({CoreRange({4, 0}, {5, 3})}),
        CoreRangeSet({CoreRange({6, 0}, {7, 3})}),
        CoreRangeSet({CoreRange({0, 4}, {3, 5})}),
        CoreRangeSet({CoreRange({4, 4}, {7, 5})}),
        CoreRangeSet({CoreRange({0, 6}, {7, 6})})
    }};

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        for (int i = 0; i < N_PARALLEL; i++) {
            auto out = ttnn::matmul(x_parts[i], w_parts[i], false, true,
                                    std::nullopt, std::nullopt, std::nullopt,
                                    std::nullopt, std::nullopt, grids[i]);
        }
        Finish(cq);
    }

    // Timed
    std::vector<double> times_part(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_PARALLEL; i++) {
            auto out = ttnn::matmul(x_parts[i], w_parts[i], false, true,
                                    std::nullopt, std::nullopt, std::nullopt,
                                    std::nullopt, std::nullopt, grids[i]);
        }
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_part[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_part = std::accumulate(times_part.begin(), times_part.end(), 0.0) / N_TIMED;
    double throughput_part = (batch_part * N_PARALLEL) / avg_part * 1000.0;

    fmt::print("## 7x Partitioned (7 matmuls × {} batch each)\n", batch_part);
    fmt::print("avg_ms,throughput_samples_sec\n");
    fmt::print("{:.3f},{:.1f}\n", avg_part, throughput_part);
    fmt::print("#\n");

    // ============================================================
    // Summary
    // ============================================================
    double speedup = throughput_part / throughput_full;
    fmt::print("## Summary\n");
    fmt::print("config,batch_total,avg_ms,throughput,speedup\n");
    fmt::print("full_grid,{},{:.3f},{:.1f},1.00x\n", batch_full, avg_full, throughput_full);
    fmt::print("7x_partitioned,{},{:.3f},{:.1f},{:.2f}x\n",
               batch_part * N_PARALLEL, avg_part, throughput_part, speedup);

    device->close();
    return 0;
}
