// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: Mesh Partitioning with Trace API
// Compares all 4 combinations:
//   1. Full Grid, NO trace
//   2. Full Grid, WITH trace
//   3. Partitioned, NO trace
//   4. Partitioned, WITH trace

#include "traced/mesh.hpp"
#include "traced/mlp.hpp"
#include "traced/trace.hpp"
#include "common.hpp"

#include <chrono>
#include <vector>
#include <numeric>

using namespace traced;
using namespace tt::tt_metal::distributed;

constexpr int N_WARMUP = 5;
constexpr int N_TIMED = 10;

int main() {
    // Create device with trace region allocated
    auto device = MeshDevice::create_unit_mesh(
        0,                          // device_id
        DEFAULT_L1_SMALL_SIZE,
        128 * 1024 * 1024,          // trace_region_size = 128MB
        1,                          // num_command_queues
        tt::tt_metal::DispatchCoreConfig{}
    );
    MeshCommandQueue& cq = device->mesh_command_queue();

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t total_cores = grid_size.x * grid_size.y;

    fmt::print("# Mesh Partitioning with Trace API Benchmark\n");
    fmt::print("# Device grid: {}x{} = {} cores\n", grid_size.x, grid_size.y, total_cores);
    fmt::print("# Warmup: {}, Timed: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");

    uint32_t batch = 512;
    uint32_t in_dim = 1024;
    uint32_t hidden_dim = 1024;
    uint32_t out_dim = 1024;
    uint32_t n_partitions = 4;
    uint32_t total_batch = batch * n_partitions;

    // ============================================================
    // Setup: Full Grid MLPs
    // ============================================================
    std::vector<TracedMLP<2>> full_mlps;
    std::vector<Tensor> full_inputs;
    for (uint32_t i = 0; i < n_partitions; i++) {
        full_mlps.emplace_back(batch, in_dim, 0.01f, *device);
        full_inputs.push_back(ttnn::ones(ttnn::Shape({batch, in_dim}),
                              ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device));
    }
    Finish(cq);

    // ============================================================
    // Setup: Partitioned MLPs
    // ============================================================
    auto mesh_mlp = TracedMeshModel<TracedPartitionedMLP>::create_mlp(
        *device, n_partitions, batch, in_dim, hidden_dim, out_dim
    );

    std::vector<Tensor> mesh_inputs;
    for (uint32_t i = 0; i < n_partitions; i++) {
        mesh_inputs.push_back(ttnn::ones(ttnn::Shape({batch, in_dim}),
                              ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device));
    }
    Finish(cq);

    // Print partition layout
    fmt::print("# Partition layout:\n");
    for (const auto& p : mesh_mlp.config.partitions) {
        auto ranges = p.core_range.ranges();
        if (!ranges.empty()) {
            auto& r = *ranges.begin();
            fmt::print("#   Partition {}: grid={}x{} ({} cores), cores=({},{}) to ({},{})\n",
                       p.partition_id, p.grid.x, p.grid.y, p.grid.x * p.grid.y,
                       r.start_coord.x, r.start_coord.y, r.end_coord.x, r.end_coord.y);
        }
    }
    fmt::print("#\n");

    // ============================================================
    // Test 1: Full Grid, NO trace
    // ============================================================
    fmt::print("## Test 1: Full Grid, NO trace\n");

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        for (uint32_t i = 0; i < n_partitions; i++) {
            full_mlps[i].forward(full_inputs[i]);
        }
        Finish(cq);
    }

    std::vector<double> times_full_notrace(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < n_partitions; i++) {
            full_mlps[i].forward(full_inputs[i]);
        }
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_full_notrace[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_full_notrace = std::accumulate(times_full_notrace.begin(), times_full_notrace.end(), 0.0) / N_TIMED;
    double tp_full_notrace = total_batch / avg_full_notrace * 1000.0;
    fmt::print("# avg_ms: {:.3f}, throughput: {:.1f} samples/sec\n", avg_full_notrace, tp_full_notrace);
    fmt::print("#\n");

    // ============================================================
    // Test 2: Full Grid, WITH trace
    // ============================================================
    fmt::print("## Test 2: Full Grid, WITH trace\n");

    TraceContext trace_full(device.get());

    // Capture trace
    trace_full.run([&]() {
        for (uint32_t i = 0; i < n_partitions; i++) {
            full_mlps[i].forward(full_inputs[i]);
        }
    });

    // Warmup with trace
    for (int w = 0; w < N_WARMUP; w++) {
        trace_full.run([&]() {
            for (uint32_t i = 0; i < n_partitions; i++) {
                full_mlps[i].forward(full_inputs[i]);
            }
        });
        Finish(cq);
    }

    std::vector<double> times_full_traced(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        trace_full.run([&]() {
            for (uint32_t i = 0; i < n_partitions; i++) {
                full_mlps[i].forward(full_inputs[i]);
            }
        });
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_full_traced[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    trace_full.release();

    double avg_full_traced = std::accumulate(times_full_traced.begin(), times_full_traced.end(), 0.0) / N_TIMED;
    double tp_full_traced = total_batch / avg_full_traced * 1000.0;
    fmt::print("# avg_ms: {:.3f}, throughput: {:.1f} samples/sec\n", avg_full_traced, tp_full_traced);
    fmt::print("#\n");

    // ============================================================
    // Test 3: Partitioned, NO trace
    // ============================================================
    fmt::print("## Test 3: Partitioned, NO trace\n");

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        mesh_mlp.forward(mesh_inputs);
        Finish(cq);
    }

    std::vector<double> times_part_notrace(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        mesh_mlp.forward(mesh_inputs);
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_part_notrace[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_part_notrace = std::accumulate(times_part_notrace.begin(), times_part_notrace.end(), 0.0) / N_TIMED;
    double tp_part_notrace = total_batch / avg_part_notrace * 1000.0;
    fmt::print("# avg_ms: {:.3f}, throughput: {:.1f} samples/sec\n", avg_part_notrace, tp_part_notrace);
    fmt::print("#\n");

    // ============================================================
    // Test 4: Partitioned, WITH trace
    // ============================================================
    fmt::print("## Test 4: Partitioned, WITH trace\n");

    TraceContext trace_part(device.get());

    // Capture trace
    trace_part.run([&]() {
        mesh_mlp.forward(mesh_inputs);
    });

    // Warmup with trace
    for (int w = 0; w < N_WARMUP; w++) {
        trace_part.run([&]() {
            mesh_mlp.forward(mesh_inputs);
        });
        Finish(cq);
    }

    std::vector<double> times_part_traced(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        trace_part.run([&]() {
            mesh_mlp.forward(mesh_inputs);
        });
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_part_traced[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    trace_part.release();

    double avg_part_traced = std::accumulate(times_part_traced.begin(), times_part_traced.end(), 0.0) / N_TIMED;
    double tp_part_traced = total_batch / avg_part_traced * 1000.0;
    fmt::print("# avg_ms: {:.3f}, throughput: {:.1f} samples/sec\n", avg_part_traced, tp_part_traced);
    fmt::print("#\n");

    // ============================================================
    // Summary
    // ============================================================
    fmt::print("## Summary\n");
    fmt::print("config,traced,batch_total,n_models,cores_per_model,avg_ms,throughput,speedup_vs_best\n");

    // Find the best throughput for normalization
    double best_tp = std::max({tp_full_notrace, tp_full_traced, tp_part_notrace, tp_part_traced});

    fmt::print("full_grid,no,{},{},{},{:.3f},{:.1f},{:.2f}x\n",
               total_batch, n_partitions, total_cores, avg_full_notrace, tp_full_notrace, tp_full_notrace / best_tp);
    fmt::print("full_grid,yes,{},{},{},{:.3f},{:.1f},{:.2f}x\n",
               total_batch, n_partitions, total_cores, avg_full_traced, tp_full_traced, tp_full_traced / best_tp);
    fmt::print("partitioned,no,{},{},{},{:.3f},{:.1f},{:.2f}x\n",
               total_batch, n_partitions, mesh_mlp.config.cores_per_partition, avg_part_notrace, tp_part_notrace, tp_part_notrace / best_tp);
    fmt::print("partitioned,yes,{},{},{},{:.3f},{:.1f},{:.2f}x\n",
               total_batch, n_partitions, mesh_mlp.config.cores_per_partition, avg_part_traced, tp_part_traced, tp_part_traced / best_tp);

    fmt::print("#\n");
    fmt::print("# Analysis:\n");
    fmt::print("#   Full grid trace speedup: {:.2f}x\n", tp_full_traced / tp_full_notrace);
    fmt::print("#   Partitioned trace speedup: {:.2f}x\n", tp_part_traced / tp_part_notrace);
    fmt::print("#   Partitioned+trace vs Full+trace: {:.2f}x\n", tp_part_traced / tp_full_traced);
    fmt::print("#   Partitioned+trace vs Full (no trace): {:.2f}x\n", tp_part_traced / tp_full_notrace);
    fmt::print("#\n");

    // Individual timings
    fmt::print("# Full grid (no trace) times: ");
    for (auto t : times_full_notrace) fmt::print("{:.3f} ", t);
    fmt::print("ms\n");
    fmt::print("# Full grid (traced) times: ");
    for (auto t : times_full_traced) fmt::print("{:.3f} ", t);
    fmt::print("ms\n");
    fmt::print("# Partitioned (no trace) times: ");
    for (auto t : times_part_notrace) fmt::print("{:.3f} ", t);
    fmt::print("ms\n");
    fmt::print("# Partitioned (traced) times: ");
    for (auto t : times_part_traced) fmt::print("{:.3f} ", t);
    fmt::print("ms\n");

    device->close();
    return 0;
}
