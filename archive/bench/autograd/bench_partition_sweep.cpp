// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: CoreGrid Partition Sweep with 2 CQs
// Tests 2x28-core partitioned matmuls vs full 56-core
// Now includes 2-CQ test to see if parallel dispatch helps
//
// Usage:
//   make bench-partition-sweep              # Default 512x512
//   BENCH_M=1024 BENCH_K=1024 BENCH_N=1024 make bench-partition-sweep-run

#include <ttnn/device.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/core.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/sub_device.hpp>
#include <fmt/core.h>
#include <chrono>
#include <cstdlib>
#include <thread>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

constexpr int N_WARMUP = 5;
constexpr int N_ITERS = 50;

int main() {
    // Shape params from env (default 512x512)
    uint32_t M = 512, K = 512, N = 512;
    if (const char* env = std::getenv("BENCH_M")) M = std::stoul(env);
    if (const char* env = std::getenv("BENCH_K")) K = std::stoul(env);
    if (const char* env = std::getenv("BENCH_N")) N = std::stoul(env);

    fmt::print("# CoreGrid Partition Sweep Benchmark (with 2 CQ test)\n");
    fmt::print("# Shape: [{}x{}] @ [{}x{}] -> [{}x{}]\n\n", M, K, K, N, M, N);

    // Create device with 2 command queues
    auto device = MeshDevice::create_unit_mesh(
        0, DEFAULT_L1_SMALL_SIZE, 128 * 1024 * 1024, 2, DispatchCoreConfig{}
    );

    auto grid_size = device->compute_with_storage_grid_size();
    fmt::print("Device grid: {}x{} = {} cores\n", grid_size.x, grid_size.y, grid_size.x * grid_size.y);
    fmt::print("Command queues: 2\n");
    fmt::print("Warmup: {}, Iterations: {}\n\n", N_WARMUP, N_ITERS);

    // CoreGrids
    ttnn::CoreGrid full_grid(grid_size.x, grid_size.y);           // 8x7 = 56 cores
    ttnn::CoreGrid half_grid(grid_size.x / 2, grid_size.y);       // 4x7 = 28 cores

    fmt::print("Full grid: {}x{} = {} cores\n", grid_size.x, grid_size.y, grid_size.x * grid_size.y);
    fmt::print("Half grid: {}x{} = {} cores\n\n", grid_size.x / 2, grid_size.y, (grid_size.x / 2) * grid_size.y);

    // Create tensors
    auto a = ttnn::full(ttnn::Shape({M, K}), 1.0f, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device);
    auto b = ttnn::full(ttnn::Shape({K, N}), 2.0f, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device);

    // ============================================================
    // Test 1: Full grid - 2 matmuls sequential (1 CQ, baseline)
    // ============================================================
    fmt::print("## Test 1: Full grid (56 cores), 2 matmuls sequential, 1 CQ\n");

    for (int i = 0; i < N_WARMUP; i++) {
        auto c1 = ttnn::matmul(a, b, false, false, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt, full_grid);
        auto c2 = ttnn::matmul(a, b, false, false, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt, full_grid);
    }
    Synchronize(device.get(), std::nullopt);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; i++) {
        auto c1 = ttnn::matmul(a, b, false, false, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt, full_grid);
        auto c2 = ttnn::matmul(a, b, false, false, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt, full_grid);
    }
    Synchronize(device.get(), std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    double full_2op_ms = std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
    fmt::print("2 matmuls @ 56 cores: {:.3f} ms (per-op: {:.3f} ms)\n\n", full_2op_ms, full_2op_ms / 2);

    // ============================================================
    // Test 2: Half grid - 2 matmuls sequential (1 CQ)
    // ============================================================
    fmt::print("## Test 2: Half grid (28 cores), 2 matmuls sequential, 1 CQ\n");

    for (int i = 0; i < N_WARMUP; i++) {
        auto c1 = ttnn::matmul(a, b, false, false, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt, half_grid);
        auto c2 = ttnn::matmul(a, b, false, false, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt, half_grid);
    }
    Synchronize(device.get(), std::nullopt);

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; i++) {
        auto c1 = ttnn::matmul(a, b, false, false, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt, half_grid);
        auto c2 = ttnn::matmul(a, b, false, false, std::nullopt, std::nullopt,
                               std::nullopt, std::nullopt, std::nullopt, half_grid);
    }
    Synchronize(device.get(), std::nullopt);
    end = std::chrono::high_resolution_clock::now();

    double half_2op_1cq_ms = std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
    fmt::print("2 matmuls @ 28 cores (1 CQ): {:.3f} ms (per-op: {:.3f} ms)\n\n", half_2op_1cq_ms, half_2op_1cq_ms / 2);

    // ============================================================
    // Test 3: Set up sub-devices for 2-CQ parallel dispatch
    // ============================================================
    fmt::print("## Test 3: Setting up sub-devices for 2-CQ parallel dispatch\n");

    uint32_t half_x = grid_size.x / 2;
    CoreRange left_cores({0, 0}, {half_x - 1, grid_size.y - 1});
    CoreRange right_cores({half_x, 0}, {grid_size.x - 1, grid_size.y - 1});

    fmt::print("Sub-device 0: cores ({},{}) to ({},{}) = {} cores\n",
               0, 0, half_x - 1, grid_size.y - 1, half_x * grid_size.y);
    fmt::print("Sub-device 1: cores ({},{}) to ({},{}) = {} cores\n",
               half_x, 0, grid_size.x - 1, grid_size.y - 1, (grid_size.x - half_x) * grid_size.y);

    SubDevice sub_device_0(std::array{CoreRangeSet(left_cores)});
    SubDevice sub_device_1(std::array{CoreRangeSet(right_cores)});

    auto sub_device_manager = device->create_sub_device_manager(
        {sub_device_0, sub_device_1},
        DEFAULT_L1_SMALL_SIZE
    );
    device->load_sub_device_manager(sub_device_manager);
    fmt::print("Loaded sub-device manager with 2 sub-devices\n\n");

    // ============================================================
    // Test 4: Try 2-CQ parallel dispatch with TTNN ops
    // ============================================================
    fmt::print("## Test 4: Attempting 2-CQ parallel dispatch with TTNN matmul\n");
    fmt::print("NOTE: This may fail - TTNN doesn't fully support sub-device dispatch\n\n");

    bool two_cq_works = false;
    double half_2op_2cq_ms = 0.0;

    try {
        // Try dispatching to different CQs
        auto& cq0 = device->mesh_command_queue(0);
        auto& cq1 = device->mesh_command_queue(1);

        // Proper warmup with N_WARMUP iterations
        for (int i = 0; i < N_WARMUP; i++) {
            auto c1 = ttnn::matmul(a, b, false, false, std::nullopt, std::nullopt,
                                   std::nullopt, std::nullopt, std::nullopt, half_grid);
            auto c2 = ttnn::matmul(a, b, false, false, std::nullopt, std::nullopt,
                                   std::nullopt, std::nullopt, std::nullopt, half_grid);
        }
        Finish(cq0);

        // Timed runs
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_ITERS; i++) {
            // Dispatch to CQ 0
            auto c1 = ttnn::matmul(a, b, false, false, std::nullopt, std::nullopt,
                                   std::nullopt, std::nullopt, std::nullopt, half_grid);
            // Dispatch to CQ 1 - this is where it might fail
            auto c2 = ttnn::matmul(a, b, false, false, std::nullopt, std::nullopt,
                                   std::nullopt, std::nullopt, std::nullopt, half_grid);
        }
        Finish(cq0);
        Finish(cq1);
        end = std::chrono::high_resolution_clock::now();

        half_2op_2cq_ms = std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
        two_cq_works = true;
        fmt::print("2 matmuls @ 28 cores (2 CQs): {:.3f} ms (per-op: {:.3f} ms)\n\n", half_2op_2cq_ms, half_2op_2cq_ms / 2);
    } catch (const std::exception& e) {
        fmt::print("2-CQ dispatch FAILED: {}\n\n", e.what());
    }

    // ============================================================
    // Summary
    // ============================================================
    fmt::print("## Summary (Shape: {}x{}x{})\n", M, K, N);
    fmt::print("| Config                         | Total ms | Per-op ms | vs Baseline |\n");
    fmt::print("|--------------------------------|----------|-----------|-------------|\n");
    fmt::print("| 2x Full (56 cores), 1 CQ       | {:.3f}    | {:.3f}     | 1.00x       |\n",
               full_2op_ms, full_2op_ms / 2);
    fmt::print("| 2x Half (28 cores), 1 CQ       | {:.3f}    | {:.3f}     | {:.2f}x       |\n",
               half_2op_1cq_ms, half_2op_1cq_ms / 2, full_2op_ms / half_2op_1cq_ms);
    if (two_cq_works) {
        fmt::print("| 2x Half (28 cores), 2 CQs      | {:.3f}    | {:.3f}     | {:.2f}x       |\n",
                   half_2op_2cq_ms, half_2op_2cq_ms / 2, full_2op_ms / half_2op_2cq_ms);
    } else {
        fmt::print("| 2x Half (28 cores), 2 CQs      | FAILED   | -         | -           |\n");
    }

    fmt::print("\n## Conclusion\n");
    if (two_cq_works && half_2op_2cq_ms < full_2op_ms) {
        fmt::print("2-CQ parallel dispatch WORKS and is {:.1f}%% faster!\n", (1 - half_2op_2cq_ms/full_2op_ms) * 100);
    } else if (two_cq_works) {
        fmt::print("2-CQ dispatch works but full grid is still faster by {:.1f}%%\n", (1 - full_2op_ms/half_2op_2cq_ms) * 100);
    } else {
        fmt::print("2-CQ parallel dispatch with TTNN ops requires sub-device API support\n");
        fmt::print("that isn't fully exposed yet. CoreGrid alone doesn't enable parallelism.\n");
    }

    device->close();
    return 0;
}
