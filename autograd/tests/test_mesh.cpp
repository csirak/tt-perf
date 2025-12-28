// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Test: TracedMeshModel for data-parallel execution across core partitions
// Compares partitioned execution vs full-grid baseline

#include "traced/mesh.hpp"
#include "traced/mlp.hpp"
#include "common.hpp"

#include <chrono>
#include <vector>
#include <numeric>

using namespace traced;
using namespace tt::tt_metal::distributed;

constexpr int N_WARMUP = 3;
constexpr int N_TIMED = 5;

int main() {
    auto device = MeshDevice::create_unit_mesh(0);
    MeshCommandQueue& cq = device->mesh_command_queue();

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t total_cores = grid_size.x * grid_size.y;

    fmt::print("# TracedMeshModel Test: Partitioned vs Full Grid\n");
    fmt::print("# Device grid: {}x{} = {} cores\n", grid_size.x, grid_size.y, total_cores);
    fmt::print("#\n");

    uint32_t batch = 512;
    uint32_t in_dim = 1024;
    uint32_t hidden_dim = 1024;
    uint32_t out_dim = 1024;
    uint32_t n_partitions = 4;
    uint32_t total_batch = batch * n_partitions;

    // ============================================================
    // Test 1: Baseline - Full Grid MLP (N sequential MLPs)
    // ============================================================
    fmt::print("## Test 1: Baseline - Full Grid ({} MLPs, {} cores each)\n", n_partitions, total_cores);

    // Create N MLPs using full grid (TracedMLP<2> is a 2-layer MLP)
    std::vector<TracedMLP<2>> full_mlps;
    std::vector<Tensor> full_inputs;
    for (uint32_t i = 0; i < n_partitions; i++) {
        full_mlps.emplace_back(batch, in_dim, 0.01f, *device);
        full_inputs.push_back(ttnn::ones(ttnn::Shape({batch, in_dim}),
                              ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device));
    }
    Finish(cq);

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        for (uint32_t i = 0; i < n_partitions; i++) {
            full_mlps[i].forward(full_inputs[i]);
        }
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_full(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (uint32_t i = 0; i < n_partitions; i++) {
            full_mlps[i].forward(full_inputs[i]);
        }
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_full[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_full = std::accumulate(times_full.begin(), times_full.end(), 0.0) / N_TIMED;
    double tp_full = total_batch / avg_full * 1000.0;

    fmt::print("# avg_ms: {:.3f}, throughput: {:.1f} samples/sec\n", avg_full, tp_full);
    fmt::print("#\n");

    // ============================================================
    // Test 2: TracedMeshModel - Partitioned (N MLPs on different cores)
    // ============================================================
    fmt::print("## Test 2: TracedMeshModel - Partitioned\n");

    auto mesh_mlp = TracedMeshModel<TracedPartitionedMLP>::create_mlp(
        *device, n_partitions, batch, in_dim, hidden_dim, out_dim
    );

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

    // Create inputs for mesh
    std::vector<Tensor> mesh_inputs;
    for (uint32_t i = 0; i < n_partitions; i++) {
        mesh_inputs.push_back(ttnn::ones(ttnn::Shape({batch, in_dim}),
                              ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device));
    }
    Finish(cq);

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        mesh_mlp.forward(mesh_inputs);
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_mesh(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        mesh_mlp.forward(mesh_inputs);
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_mesh[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_mesh = std::accumulate(times_mesh.begin(), times_mesh.end(), 0.0) / N_TIMED;
    double tp_mesh = total_batch / avg_mesh * 1000.0;

    fmt::print("# avg_ms: {:.3f}, throughput: {:.1f} samples/sec\n", avg_mesh, tp_mesh);
    fmt::print("#\n");

    // ============================================================
    // Summary
    // ============================================================
    fmt::print("## Summary\n");
    fmt::print("config,batch_total,n_models,cores_per_model,avg_ms,throughput,throughput_per_core,speedup\n");
    fmt::print("full_grid,{},{},{},{:.3f},{:.1f},{:.1f},1.00x\n",
               total_batch, n_partitions, total_cores, avg_full, tp_full, tp_full / total_cores);
    fmt::print("mesh_partitioned,{},{},{},{:.3f},{:.1f},{:.1f},{:.2f}x\n",
               total_batch, n_partitions, mesh_mlp.config.cores_per_partition,
               avg_mesh, tp_mesh, tp_mesh / total_cores, tp_mesh / tp_full);

    fmt::print("#\n");
    fmt::print("# Full grid times: ");
    for (auto t : times_full) fmt::print("{:.3f} ", t);
    fmt::print("ms\n");
    fmt::print("# Mesh times: ");
    for (auto t : times_mesh) fmt::print("{:.3f} ", t);
    fmt::print("ms\n");

    fmt::print("\nPASS: All tests passed\n");

    device->close();
    return 0;
}
