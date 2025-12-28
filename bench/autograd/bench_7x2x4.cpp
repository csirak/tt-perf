// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// 7x Parallel 2x4 Core Groups Benchmark
// Tests data parallel scaling with 7 independent MLPs, each using 8 cores.
//
// Wormhole ASIC Tensix core layout (8 cols × 7 rows = 56 cores):
//   Columns: 1-4, 6-9 (skip 0=DRAM/PCIe, 5=DRAM)
//   Rows: 1-5, 7-11 (skip 0=Ethernet, 6=Ethernet)
//   But compute_with_storage_grid reports 8×7, so logical coords are 0-7, 0-6
//
// Config: 7 MLPs, each using 2x4=8 cores for ALL ops
// Total: 7 × 8 = 56 cores (full chip utilization)

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

// Helper type alias for empty activation spans
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

// Forward + backward with ALL ops constrained to specific cores
void run_mlp_async(const MLPInstance& m, const CoreGrid& grid, const CoreRangeSet& core_range) {
    // Forward: 5 ops - ALL constrained to core_range
    auto mm1 = ttnn::matmul(m.x, m.w1, false, true, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, std::nullopt, grid);

    auto h_pre = ttnn::add(mm1, m.b1,
                           std::nullopt, std::nullopt, std::nullopt,
                           ActivationSpan{}, ActivationSpan{}, ActivationSpan{},
                           std::nullopt, core_range);

    auto h = ttnn::relu(h_pre, std::nullopt, std::nullopt, core_range);

    auto mm2 = ttnn::matmul(h, m.w2, false, true, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, std::nullopt, grid);

    auto out = ttnn::add(mm2, m.b2, std::nullopt, std::nullopt, std::nullopt,
                         ActivationSpan{}, ActivationSpan{}, ActivationSpan{},
                         std::nullopt, core_range);

    // Backward: 5 ops - ALL constrained to core_range
    auto dh = ttnn::matmul(m.dout, m.w2, false, false, std::nullopt, std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, grid);

    auto relu_mask = ttnn::gtz(h_pre, std::nullopt, std::nullopt, core_range);

    auto dh_pre = ttnn::multiply(dh, relu_mask,
                                  std::nullopt, std::nullopt, std::nullopt,
                                  ActivationSpan{}, ActivationSpan{}, ActivationSpan{},
                                  std::nullopt, core_range);

    auto dx = ttnn::matmul(dh_pre, m.w1, false, false, std::nullopt, std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, grid);
}

int main() {
    auto device = MeshDevice::create_unit_mesh(0);
    MeshCommandQueue& cq = device->mesh_command_queue();

    auto grid_size = device->compute_with_storage_grid_size();
    fmt::print("# 7x Parallel 8-Core Groups Benchmark\n");
    fmt::print("# Device grid: {}x{} = {} cores (logical)\n", grid_size.x, grid_size.y,
               grid_size.x * grid_size.y);
    fmt::print("# Config: {} MLPs, each using 8 cores (various layouts)\n", N_PARALLEL);
    fmt::print("# Total cores used: {} × 8 = {} cores\n", N_PARALLEL, N_PARALLEL * 8);
    fmt::print("# Warmup: {}, Timed: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");

    // Define 7 non-overlapping 2x4 core regions
    // Wormhole grid is 8x7 (columns 0-9 with gaps, rows 1-9 with gaps)
    // Physical layout (avoiding column 0 and 5 which are special):
    //   Group 0: cols 1-2, rows 1-4  -> (1,1)-(2,4)
    //   Group 1: cols 3-4, rows 1-4  -> (3,1)-(4,4)
    //   Group 2: cols 6-7, rows 1-4  -> (6,1)-(7,4)
    //   Group 3: cols 8-9, rows 1-4  -> (8,1)-(9,4)
    //   Group 4: cols 1-2, rows 5,7  -> (1,5)-(2,7) but skip row 6
    //   Group 5: cols 3-4, rows 5,7  -> (3,5)-(4,7)
    //   Group 6: cols 6-7, rows 5,7  -> (6,5)-(7,7)

    // Actually, let's use contiguous 2x4 blocks more carefully
    // The compute grid is 8 columns x 7 rows = 56 cores
    // We need 7 groups of 8 cores each

    // TTNN uses LOGICAL coordinates: 8 cols (0-7) × 7 rows (0-6) = 56 cores
    // These map to physical Tensix cores automatically.
    //
    // Layout: 7 groups of 8 cores each
    // Groups 0-3: 2 cols × 4 rows in top 4 rows (rows 0-3)
    // Groups 4-6: different arrangements using remaining 3 rows (4-6)
    //
    // Top 4 rows (0-3): 8 cols × 4 rows = 32 cores = 4 groups of 8
    // Bottom 3 rows (4-6): 8 cols × 3 rows = 24 cores = 3 groups of 8

    std::array<CoreGrid, N_PARALLEL> grids = {{
        CoreGrid(2, 4), CoreGrid(2, 4), CoreGrid(2, 4), CoreGrid(2, 4),
        CoreGrid(4, 2), CoreGrid(4, 2), CoreGrid(8, 1)  // Last one is 8×1 for row 6
    }};

    // LOGICAL coordinates (0-indexed)
    std::array<CoreRangeSet, N_PARALLEL> core_ranges = {{
        // Top half: 4 groups of 2×4 = 8 cores each
        CoreRangeSet({CoreRange({0, 0}, {1, 3})}),  // cols 0-1, rows 0-3
        CoreRangeSet({CoreRange({2, 0}, {3, 3})}),  // cols 2-3, rows 0-3
        CoreRangeSet({CoreRange({4, 0}, {5, 3})}),  // cols 4-5, rows 0-3
        CoreRangeSet({CoreRange({6, 0}, {7, 3})}),  // cols 6-7, rows 0-3
        // Bottom: 3 groups using rows 4-6
        CoreRangeSet({CoreRange({0, 4}, {3, 5})}),  // cols 0-3, rows 4-5 = 4×2 = 8
        CoreRangeSet({CoreRange({4, 4}, {7, 5})}),  // cols 4-7, rows 4-5 = 4×2 = 8
        CoreRangeSet({CoreRange({0, 6}, {7, 6})})   // cols 0-7, row 6 = 8×1 = 8
    }};

    // Smaller batch for 7 parallel MLPs (divide work)
    uint32_t batch = 64;  // Smaller batch per MLP
    uint32_t dim = 256;   // Smaller dim for 8 cores

    // Create 7 MLPs
    std::array<MLPInstance, N_PARALLEL> mlps;
    for (int i = 0; i < N_PARALLEL; i++) {
        mlps[i] = create_mlp(*device, batch, dim);
    }
    Finish(cq);

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
    double throughput_per_core = throughput / (N_PARALLEL * 8);

    fmt::print("batch,dim,n_parallel,cores_per_mlp,total_cores,avg_ms,throughput_samples_sec,throughput_per_core\n");
    fmt::print("{},{},{},8,{},{:.3f},{:.1f},{:.1f}\n",
               batch, dim, N_PARALLEL, N_PARALLEL * 8, avg_ms, throughput, throughput_per_core);

    fmt::print("#\n");
    fmt::print("# Individual run times: ");
    for (auto t : times_ms) fmt::print("{:.3f} ", t);
    fmt::print("ms\n");

    device->close();
    return 0;
}
