// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// 7x Parallel Uniform CoreGrid Benchmark
// Tests maximum data parallel throughput with 7 MLPs.
//
// Wormhole 8x7 logical grid = 56 cores
// Config: 7 MLPs, ALL using uniform CoreGrid(2,4) for matmuls
//
// Layout strategy:
// - Top 4 rows (0-3): 4 groups of 2×4 = 32 cores
// - Bottom 3 rows (4-6): 3 groups of 2×4 spanning rows creatively
//   Since we only have 3 rows, we use 2 cols × 4 rows but with
//   rows wrapping or using adjacent columns.
//
// Actually: Use CoreGrid(2,4) uniformly but with CoreRangeSets
// that may use 6-8 cores depending on available space.

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>

#include <chrono>
#include <vector>
#include <numeric>
#include <array>

using namespace ttnn;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

using ActivationSpan = tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam>;

constexpr int N_WARMUP = 2;
constexpr int N_TIMED = 3;
constexpr int N_PARALLEL = 7;

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

void run_mlp_async(const MLPInstance& m, const CoreGrid& grid, const CoreRangeSet& core_range) {
    // Forward
    auto mm1 = ttnn::matmul(m.x, m.w1, false, true, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, std::nullopt, grid);
    auto h_pre = ttnn::add(mm1, m.b1, std::nullopt, std::nullopt, std::nullopt,
                           ActivationSpan{}, ActivationSpan{}, ActivationSpan{},
                           std::nullopt, core_range);
    auto h = ttnn::relu(h_pre, std::nullopt, std::nullopt, core_range);
    auto mm2 = ttnn::matmul(h, m.w2, false, true, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, std::nullopt, grid);
    auto out = ttnn::add(mm2, m.b2, std::nullopt, std::nullopt, std::nullopt,
                         ActivationSpan{}, ActivationSpan{}, ActivationSpan{},
                         std::nullopt, core_range);

    // Backward
    auto dh = ttnn::matmul(m.dout, m.w2, false, false, std::nullopt, std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, grid);
    auto relu_mask = ttnn::gtz(h_pre, std::nullopt, std::nullopt, core_range);
    auto dh_pre = ttnn::multiply(dh, relu_mask, std::nullopt, std::nullopt, std::nullopt,
                                  ActivationSpan{}, ActivationSpan{}, ActivationSpan{},
                                  std::nullopt, core_range);
    auto dx = ttnn::matmul(dh_pre, m.w1, false, false, std::nullopt, std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, grid);
}

int main() {
    auto device = MeshDevice::create_unit_mesh(0);
    MeshCommandQueue& cq = device->mesh_command_queue();

    auto grid_size = device->compute_with_storage_grid_size();
    fmt::print("# 7x Parallel Uniform CoreGrid Benchmark\n");
    fmt::print("# Device grid: {}x{} = {} cores (logical)\n", grid_size.x, grid_size.y,
               grid_size.x * grid_size.y);
    fmt::print("# Config: {} MLPs, all using uniform CoreGrid(2,4)\n", N_PARALLEL);
    fmt::print("# Warmup: {}, Timed: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");

    // ALL grids uniform 2×4 (this is the key difference from bench_7x2x4.cpp)
    std::array<CoreGrid, N_PARALLEL> grids = {{
        CoreGrid(2, 4), CoreGrid(2, 4), CoreGrid(2, 4), CoreGrid(2, 4),
        CoreGrid(2, 4), CoreGrid(2, 4), CoreGrid(2, 4)
    }};

    // Layout: 7 regions, 56 cores total
    // Top half (rows 0-3): 4 groups of 2×4 = 32 cores
    // Bottom half (rows 4-6): 3 groups
    //   - Groups 4,5: 2 cols × 3 rows = 6 cores each (rows 4-6)
    //   - Group 6: 2 cols × 3 rows = 6 cores (rows 4-6)
    //   But we want 8 cores each, so let's be creative:
    //   Use multi-range CoreRangeSets to get 8 cores per group
    //
    // Alternative layout for bottom (24 cores in rows 4-6):
    //   Group 4: cols 0-1, rows 4-6 (6 cores) + cols 0-1, row 0 (2 cores from top)
    //   NO - that would overlap with group 0
    //
    // Actual layout: Accept 6 cores for bottom groups
    // Total: 4*8 + 3*6 = 32 + 18 = 50 cores (6 unused)
    //
    // OR: Reorganize to use all 56 cores with 8 each:
    //   Use 7 × 8 = 56 cores with varied rectangular shapes
    //   Keep CoreGrid(2,4) uniform, but CoreRangeSet varies
    //
    // Final layout - maximize 8-core groups:
    //   Groups 0-3: rows 0-3, cols 0-1, 2-3, 4-5, 6-7 (4 × 8 = 32)
    //   Groups 4-6: Use remaining 24 cores in rows 4-6 as 3 × 8
    //     - This requires 4×2 or 8×1 shapes, but we keep CoreGrid(2,4)
    //     - CoreRangeSet for group 4: cols 0-3, rows 4-5 (8 cores)
    //     - CoreRangeSet for group 5: cols 4-7, rows 4-5 (8 cores)
    //     - CoreRangeSet for group 6: cols 0-7, row 6 (8 cores)
    //
    // Key insight: CoreGrid controls matmul parallelization strategy
    // CoreRangeSet controls which cores the op runs on
    // Using CoreGrid(2,4) with a 4×2 CoreRangeSet should still work

    std::array<CoreRangeSet, N_PARALLEL> core_ranges = {{
        // Top half: 4 groups of 2×4
        CoreRangeSet({CoreRange({0, 0}, {1, 3})}),  // cols 0-1, rows 0-3 = 8 cores
        CoreRangeSet({CoreRange({2, 0}, {3, 3})}),  // cols 2-3, rows 0-3 = 8 cores
        CoreRangeSet({CoreRange({4, 0}, {5, 3})}),  // cols 4-5, rows 0-3 = 8 cores
        CoreRangeSet({CoreRange({6, 0}, {7, 3})}),  // cols 6-7, rows 0-3 = 8 cores
        // Bottom half: 3 groups of 8 (different shapes)
        CoreRangeSet({CoreRange({0, 4}, {3, 5})}),  // cols 0-3, rows 4-5 = 8 cores
        CoreRangeSet({CoreRange({4, 4}, {7, 5})}),  // cols 4-7, rows 4-5 = 8 cores
        CoreRangeSet({CoreRange({0, 6}, {7, 6})})   // cols 0-7, row 6 = 8 cores
    }};

    // Larger sizes to fully utilize all cores
    uint32_t batch = 256;
    uint32_t dim = 512;

    // Create 7 MLPs
    std::array<MLPInstance, N_PARALLEL> mlps;
    for (int i = 0; i < N_PARALLEL; i++) {
        mlps[i] = create_mlp(*device, batch, dim);
    }
    Finish(cq);

    fmt::print("# Core layout:\n");
    fmt::print("#   Groups 0-3: 2×4 vertical strips in rows 0-3\n");
    fmt::print("#   Groups 4-5: 4×2 horizontal strips in rows 4-5\n");
    fmt::print("#   Group 6: 8×1 horizontal strip in row 6\n");
    fmt::print("#   ALL use CoreGrid(2,4) for uniform matmul strategy\n");
    fmt::print("#\n");

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        for (int i = 0; i < N_PARALLEL; i++) {
            run_mlp_async(mlps[i], grids[i], core_ranges[i]);
        }
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_PARALLEL; i++) {
            run_mlp_async(mlps[i], grids[i], core_ranges[i]);
        }
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
    double total_samples = N_PARALLEL * batch;
    double throughput = total_samples / avg_ms * 1000.0;
    double throughput_per_core = throughput / 56;  // Using all 56 cores

    fmt::print("batch,dim,n_parallel,cores_per_mlp,total_cores,avg_ms,throughput_samples_sec,throughput_per_core\n");
    fmt::print("{},{},{},8,56,{:.3f},{:.1f},{:.1f}\n",
               batch, dim, N_PARALLEL, avg_ms, throughput, throughput_per_core);

    fmt::print("#\n");
    fmt::print("# Individual run times: ");
    for (auto t : times_ms) fmt::print("{:.3f} ", t);
    fmt::print("ms\n");

    device->close();
    return 0;
}
