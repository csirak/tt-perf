// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Traced Matmul Comparison: Full Grid vs 7x Partitioned
// Compares throughput with and without Trace API to isolate dispatch overhead
//
// Tests:
// 1. Full grid, no trace (baseline)
// 2. Full grid, WITH trace (fair comparison)
// 3. 7x partitioned, no trace (dispatch overhead)
// 4. 7x partitioned, WITH trace (should match full grid)

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/trace.hpp>
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
constexpr int N_TIMED = 10;
constexpr int N_PARALLEL = 7;

int main() {
    // Create device with trace region allocated (required for trace API)
    auto device = MeshDevice::create_unit_mesh(
        0,                          // device_id
        DEFAULT_L1_SMALL_SIZE,
        128 * 1024 * 1024,          // trace_region_size = 128MB
        1,                          // num_command_queues
        DispatchCoreConfig{}
    );
    MeshCommandQueue& cq = device->mesh_command_queue();

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t total_cores = grid_size.x * grid_size.y;

    fmt::print("# Traced Matmul Comparison: Full Grid vs 7x Partitioned\n");
    fmt::print("# Device grid: {}x{} = {} cores\n", grid_size.x, grid_size.y, total_cores);
    fmt::print("# Warmup: {}, Timed: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");

    uint32_t batch_part = 256;
    uint32_t dim = 512;
    uint32_t batch_full = batch_part * N_PARALLEL;  // 1792

    // ============================================================
    // Test 1: Full grid, NO trace (baseline)
    // ============================================================
    auto x_full = ones(Shape({batch_full, dim}), DataType::BFLOAT16, TILE_LAYOUT, *device);
    auto w_full = ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, *device);
    Finish(cq);

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        auto out = ttnn::matmul(x_full, w_full);
        Finish(cq);
    }

    // Timed
    std::vector<double> times_full_notrace(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto out = ttnn::matmul(x_full, w_full);
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_full_notrace[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_full_notrace = std::accumulate(times_full_notrace.begin(), times_full_notrace.end(), 0.0) / N_TIMED;
    double tp_full_notrace = batch_full / avg_full_notrace * 1000.0;

    fmt::print("## Test 1: Full Grid, NO trace\n");
    fmt::print("avg_ms,throughput_samples_sec\n");
    fmt::print("{:.3f},{:.1f}\n", avg_full_notrace, tp_full_notrace);
    fmt::print("#\n");

    // ============================================================
    // Test 2: Full grid, WITH trace
    // ============================================================
    Tensor out_full_traced;  // Pre-allocate output reference

    // Capture trace
    auto trace_full = ttnn::operations::trace::begin_trace_capture(device.get(), std::nullopt);
    out_full_traced = ttnn::matmul(x_full, w_full);
    ttnn::operations::trace::end_trace_capture(device.get(), trace_full, std::nullopt);

    // Warmup with trace
    for (int w = 0; w < N_WARMUP; w++) {
        ttnn::operations::trace::execute_trace(device.get(), trace_full, std::nullopt, false);
        Finish(cq);
    }

    // Timed with trace
    std::vector<double> times_full_traced(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        ttnn::operations::trace::execute_trace(device.get(), trace_full, std::nullopt, false);
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_full_traced[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    ttnn::operations::trace::release_trace(device.get(), trace_full);

    double avg_full_traced = std::accumulate(times_full_traced.begin(), times_full_traced.end(), 0.0) / N_TIMED;
    double tp_full_traced = batch_full / avg_full_traced * 1000.0;

    fmt::print("## Test 2: Full Grid, WITH trace\n");
    fmt::print("avg_ms,throughput_samples_sec\n");
    fmt::print("{:.3f},{:.1f}\n", avg_full_traced, tp_full_traced);
    fmt::print("#\n");

    // ============================================================
    // Test 3: 7x Partitioned, NO trace
    // ============================================================
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
    std::vector<double> times_part_notrace(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_PARALLEL; i++) {
            auto out = ttnn::matmul(x_parts[i], w_parts[i], false, true,
                                    std::nullopt, std::nullopt, std::nullopt,
                                    std::nullopt, std::nullopt, grids[i]);
        }
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_part_notrace[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_part_notrace = std::accumulate(times_part_notrace.begin(), times_part_notrace.end(), 0.0) / N_TIMED;
    double tp_part_notrace = batch_full / avg_part_notrace * 1000.0;

    fmt::print("## Test 3: 7x Partitioned, NO trace\n");
    fmt::print("avg_ms,throughput_samples_sec\n");
    fmt::print("{:.3f},{:.1f}\n", avg_part_notrace, tp_part_notrace);
    fmt::print("#\n");

    // ============================================================
    // Test 4: 7x Partitioned, WITH trace
    // ============================================================
    std::array<Tensor, N_PARALLEL> out_parts_traced;  // Pre-allocate output references

    // Capture trace of all 7 matmuls
    auto trace_part = ttnn::operations::trace::begin_trace_capture(device.get(), std::nullopt);
    for (int i = 0; i < N_PARALLEL; i++) {
        out_parts_traced[i] = ttnn::matmul(x_parts[i], w_parts[i], false, true,
                                           std::nullopt, std::nullopt, std::nullopt,
                                           std::nullopt, std::nullopt, grids[i]);
    }
    ttnn::operations::trace::end_trace_capture(device.get(), trace_part, std::nullopt);

    // Warmup with trace
    for (int w = 0; w < N_WARMUP; w++) {
        ttnn::operations::trace::execute_trace(device.get(), trace_part, std::nullopt, false);
        Finish(cq);
    }

    // Timed with trace
    std::vector<double> times_part_traced(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        ttnn::operations::trace::execute_trace(device.get(), trace_part, std::nullopt, false);
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_part_traced[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    ttnn::operations::trace::release_trace(device.get(), trace_part);

    double avg_part_traced = std::accumulate(times_part_traced.begin(), times_part_traced.end(), 0.0) / N_TIMED;
    double tp_part_traced = batch_full / avg_part_traced * 1000.0;

    fmt::print("## Test 4: 7x Partitioned, WITH trace\n");
    fmt::print("avg_ms,throughput_samples_sec\n");
    fmt::print("{:.3f},{:.1f}\n", avg_part_traced, tp_part_traced);
    fmt::print("#\n");

    // ============================================================
    // Summary
    // ============================================================
    fmt::print("## Summary\n");
    fmt::print("config,traced,batch_total,avg_ms,throughput,speedup_vs_full_traced\n");
    fmt::print("full_grid,no,{},{:.3f},{:.1f},{:.2f}x\n",
               batch_full, avg_full_notrace, tp_full_notrace, tp_full_notrace / tp_full_traced);
    fmt::print("full_grid,yes,{},{:.3f},{:.1f},{:.2f}x\n",
               batch_full, avg_full_traced, tp_full_traced, 1.0);
    fmt::print("7x_partitioned,no,{},{:.3f},{:.1f},{:.2f}x\n",
               batch_full, avg_part_notrace, tp_part_notrace, tp_part_notrace / tp_full_traced);
    fmt::print("7x_partitioned,yes,{},{:.3f},{:.1f},{:.2f}x\n",
               batch_full, avg_part_traced, tp_part_traced, tp_part_traced / tp_full_traced);

    fmt::print("#\n");
    fmt::print("# Dispatch overhead analysis:\n");
    fmt::print("#   Full grid trace speedup: {:.2f}x\n", tp_full_traced / tp_full_notrace);
    fmt::print("#   Partitioned trace speedup: {:.2f}x\n", tp_part_traced / tp_part_notrace);
    fmt::print("#   Partitioned vs Full (both traced): {:.2f}x\n", tp_part_traced / tp_full_traced);

    device->close();
    return 0;
}
