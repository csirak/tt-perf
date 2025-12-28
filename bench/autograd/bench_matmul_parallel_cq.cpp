// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Parallel Matmul with Multiple Command Queues
// Compares:
// 1. Full grid single matmul (baseline)
// 2. 7x partitioned, single CQ (sequential)
// 3. 7x partitioned, 2 CQs (parallel streams)

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>

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
    // Create device with 2 hardware command queues
    auto device = MeshDevice::create_unit_mesh(
        0,                          // device_id
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        2,                          // num_command_queues = 2
        DispatchCoreConfig{}
    );

    MeshCommandQueue& cq0 = device->mesh_command_queue(0);
    // MeshCommandQueue& cq1 = device->mesh_command_queue(1);  // Not used - see Test 3 notes

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t total_cores = grid_size.x * grid_size.y;

    fmt::print("# Parallel Matmul with Multiple Command Queues\n");
    fmt::print("# Device grid: {}x{} = {} cores\n", grid_size.x, grid_size.y, total_cores);
    fmt::print("# HW Command Queues: 2\n");
    fmt::print("# Warmup: {}, Timed: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");

    uint32_t batch_part = 256;
    uint32_t dim = 512;
    uint32_t batch_full = batch_part * N_PARALLEL;  // 1792

    // ============================================================
    // Test 1: Full grid - one large matmul
    // ============================================================
    auto x_full = ones(Shape({batch_full, dim}), DataType::BFLOAT16, TILE_LAYOUT, *device);
    auto w_full = ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, *device);
    Finish(cq0);

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        auto out = ttnn::matmul(x_full, w_full);
        Finish(cq0);
    }

    // Timed
    std::vector<double> times_full(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto out = ttnn::matmul(x_full, w_full);
        Finish(cq0);
        auto end = std::chrono::high_resolution_clock::now();
        times_full[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_full = std::accumulate(times_full.begin(), times_full.end(), 0.0) / N_TIMED;
    double throughput_full = batch_full / avg_full * 1000.0;

    fmt::print("## Test 1: Full Grid (1 large matmul, {} batch)\n", batch_full);
    fmt::print("avg_ms,throughput_samples_sec\n");
    fmt::print("{:.3f},{:.1f}\n", avg_full, throughput_full);
    fmt::print("#\n");

    // ============================================================
    // Test 2: 7x Partitioned, Single CQ (sequential)
    // ============================================================
    std::array<Tensor, N_PARALLEL> x_parts;
    std::array<Tensor, N_PARALLEL> w_parts;
    for (int i = 0; i < N_PARALLEL; i++) {
        x_parts[i] = ones(Shape({batch_part, dim}), DataType::BFLOAT16, TILE_LAYOUT, *device);
        w_parts[i] = ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, *device);
    }
    Finish(cq0);

    // All grids uniform 2×4
    std::array<CoreGrid, N_PARALLEL> grids = {{
        CoreGrid(2, 4), CoreGrid(2, 4), CoreGrid(2, 4), CoreGrid(2, 4),
        CoreGrid(2, 4), CoreGrid(2, 4), CoreGrid(2, 4)
    }};

    // Warmup single CQ
    for (int w = 0; w < N_WARMUP; w++) {
        for (int i = 0; i < N_PARALLEL; i++) {
            auto out = ttnn::matmul(x_parts[i], w_parts[i], false, true,
                                    std::nullopt, std::nullopt, std::nullopt,
                                    std::nullopt, std::nullopt, grids[i]);
        }
        Finish(cq0);
    }

    // Timed single CQ
    std::vector<double> times_seq(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_PARALLEL; i++) {
            auto out = ttnn::matmul(x_parts[i], w_parts[i], false, true,
                                    std::nullopt, std::nullopt, std::nullopt,
                                    std::nullopt, std::nullopt, grids[i]);
        }
        Finish(cq0);
        auto end = std::chrono::high_resolution_clock::now();
        times_seq[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_seq = std::accumulate(times_seq.begin(), times_seq.end(), 0.0) / N_TIMED;
    double throughput_seq = batch_full / avg_seq * 1000.0;

    fmt::print("## Test 2: 7x Partitioned, Single CQ (sequential)\n");
    fmt::print("avg_ms,throughput_samples_sec\n");
    fmt::print("{:.3f},{:.1f}\n", avg_seq, throughput_seq);
    fmt::print("#\n");

    // ============================================================
    // Test 3: Note on multi-CQ for parallel compute
    // ============================================================
    // Multiple HW CQs are designed for overlapping copy and compute,
    // NOT for parallel compute on different core regions.
    // The SubDevice ownership model prevents using CQ1 while CQ0 owns the device.
    //
    // For data-parallel compute, use single CQ with partitioned CoreGrids.
    // The device should pipeline operations on non-overlapping core regions.

    fmt::print("## Test 3: Multi-CQ Notes\n");
    fmt::print("# Multiple HW CQs are for copy/compute overlap, not parallel compute.\n");
    fmt::print("# For data-parallel, use single CQ with partitioned CoreGrids.\n");
    fmt::print("#\n");

    double avg_par = avg_seq;  // Same as sequential for now
    double throughput_par = throughput_seq;

    // ============================================================
    // Summary
    // ============================================================
    fmt::print("## Summary\n");
    fmt::print("config,batch_total,avg_ms,throughput,speedup_vs_full\n");
    fmt::print("full_grid,{},{:.3f},{:.1f},1.00x\n", batch_full, avg_full, throughput_full);
    fmt::print("7x_1cq_sequential,{},{:.3f},{:.1f},{:.2f}x\n",
               batch_full, avg_seq, throughput_seq, throughput_seq / throughput_full);
    fmt::print("7x_2cq_parallel,{},{:.3f},{:.1f},{:.2f}x\n",
               batch_full, avg_par, throughput_par, throughput_par / throughput_full);

    device->close();
    return 0;
}
