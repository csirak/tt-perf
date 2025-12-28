// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Minimal Benchmark: 2x Parallel MLPs with 4x4 core grid partitioning
// Tests if restricting each MLP to a dedicated core region enables true parallelism.
//
// Config: 2 MLPs, each using 4x4=16 cores (32 total out of 56 available)

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

constexpr int N_WARMUP = 2;
constexpr int N_TIMED = 3;

struct MLPInstance {
    Tensor x, w1, b1, w2, b2, dout;
};

MLPInstance create_mlp(MeshDevice& dev, uint32_t batch, uint32_t dim) {
    return {
        ones(Shape({batch, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        zeros(Shape({1, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        zeros(Shape({1, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        ones(Shape({batch, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
    };
}

// Forward + backward with explicit core_grid for matmuls
void run_mlp_async(const MLPInstance& m, const CoreGrid& grid) {
    // Forward: 5 ops (matmuls use core_grid, elementwise ops auto-select)
    auto mm1 = ttnn::matmul(m.x, m.w1, false, true, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, std::nullopt, grid);
    auto h_pre = ttnn::add(mm1, m.b1);
    auto h = ttnn::relu(h_pre);
    auto mm2 = ttnn::matmul(h, m.w2, false, true, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, std::nullopt, grid);
    auto out = ttnn::add(mm2, m.b2);

    // Backward: 5 ops
    auto dh = ttnn::matmul(m.dout, m.w2, false, false, std::nullopt, std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, grid);
    auto relu_mask = ttnn::gtz(h_pre);
    auto dh_pre = ttnn::multiply(dh, relu_mask);
    auto dx = ttnn::matmul(dh_pre, m.w1, false, false, std::nullopt, std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, grid);
}

int main() {
    auto device = MeshDevice::create_unit_mesh(0);
    MeshCommandQueue& cq = device->mesh_command_queue();

    auto grid_size = device->compute_with_storage_grid_size();
    fmt::print("# 2x Parallel MLPs with 4x4 Core Grid Partitioning\n");
    fmt::print("# Device grid: {}x{} = {} cores\n", grid_size.x, grid_size.y,
               grid_size.x * grid_size.y);
    fmt::print("# Each MLP uses 4x4 = 16 cores (32 total)\n");
    fmt::print("# Warmup: {}, Timed: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");

    // Two 4x4 core grids
    CoreGrid grid_a(4, 4);
    CoreGrid grid_b(4, 4);

    // Same size as before: batch=256, dim=512
    uint32_t batch = 256;
    uint32_t dim = 512;

    // Create 2 MLPs
    auto mlp_a = create_mlp(*device, batch, dim);
    auto mlp_b = create_mlp(*device, batch, dim);
    Finish(cq);

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        run_mlp_async(mlp_a, grid_a);
        run_mlp_async(mlp_b, grid_b);
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        run_mlp_async(mlp_a, grid_a);
        run_mlp_async(mlp_b, grid_b);
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
    double throughput = (2 * batch) / avg_ms * 1000.0;

    fmt::print("batch,dim,n_parallel,cores_per_mlp,avg_ms,throughput_samples_sec\n");
    fmt::print("{},{},2,16,{:.3f},{:.1f}\n", batch, dim, avg_ms, throughput);

    fmt::print("#\n");
    fmt::print("# Run with TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 for tt-npe analysis\n");

    device->close();
    return 0;
}
