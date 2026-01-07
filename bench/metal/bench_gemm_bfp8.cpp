// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GEMM Top 10 Benchmark - BFLOAT8_B with HiFi2
// Matching reference: test_benchmark.py matmul_shapes_bfloat8_b
//
// Key settings:
// - DataType::BFLOAT8_B (1 byte per element)
// - MathFidelity::HiFi2 (32 cycles/tile)
// - packer_l1_acc = true
// - ETH dispatch for 8x8 grid
// - L1 sharding for smaller shapes, DRAM for 4096x4096x4096

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/matmul/device/matmul_op.hpp>
#include <ttnn/operations/trace.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/data_movement/common/common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tile.hpp>

#include <chrono>
#include <vector>
#include <cstdlib>
#include <tuple>

using namespace ttnn;
using namespace tt::tt_metal::distributed;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;
using DispatchCoreType = tt::tt_metal::DispatchCoreType;
using CoreRange = tt::tt_metal::CoreRange;
using CoreRangeSet = tt::tt_metal::CoreRangeSet;
using ShardStrategy = ttnn::operations::data_movement::ShardStrategy;
using MatmulProgramConfig = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig;

constexpr int N_WARMUP = 5;
constexpr int N_ITER = 100;
constexpr size_t TRACE_REGION_SIZE = 128 * 1024 * 1024;

// Subblock selection priority (from test_benchmark.py)
const std::vector<std::pair<int, int>> SUBBLOCK_HW_CHOICES = {
    {4, 2}, {2, 4}, {8, 1}, {1, 8},
    {7, 1}, {1, 7},
    {3, 2}, {2, 3}, {6, 1}, {1, 6},
    {5, 1}, {1, 5},
    {2, 2}, {4, 1}, {1, 4},
    {3, 1}, {1, 3},
    {2, 1}, {1, 2},
    {1, 1},
};

std::pair<int, int> get_subblock_sizes(int m_tiles, int n_tiles, bool out_sharded = false) {
    for (auto [h, w] : SUBBLOCK_HW_CHOICES) {
        if (out_sharded && (n_tiles % w != 0 || h != 1)) continue;
        if (m_tiles % h == 0 && n_tiles % w == 0) return {h, w};
    }
    return {1, 1};
}

struct GemmConfig {
    uint32_t M, K, N;
    bool use_trace;
    double target_tflops;
    // Reference benchmark params
    bool in0_sharded;
    bool out_sharded;
    int in0_block_w_div;
    int num_out_blocks_h;
    int num_out_blocks_w;
};

// Top 10 BFP8 configs from GEMM_FLOPS.md with reference benchmark params
// Note: 3072x4096x4096 derived from BF16 list (not in BFP8 list)
std::vector<GemmConfig> TOP10_CONFIGS = {
    // M, K, N, trace, target, in0_sharded, out_sharded, in0_block_w_div, num_out_blocks_h, num_out_blocks_w
    {4096, 4096, 4096, false, 146.02, false, false, 1, 2, 2},  // DRAM
    {4096, 4096, 4096, true,  141.93, false, false, 1, 2, 2},  // DRAM
    {3072, 4096, 4096, false, 140.23, false, false, 2, 1, 1},  // DRAM (derived from BF16)
    {3072, 4096, 4096, true,  140.73, false, false, 2, 1, 1},  // DRAM (derived from BF16)
    {3072, 3072, 4096, false, 133.01, true,  true,  2, 1, 1},  // L1 sharded
    {3072, 3072, 4096, true,  133.64, true,  true,  2, 1, 1},  // L1 sharded
    {3072, 3072, 3072, false, 130.98, true,  true,  2, 1, 1},  // L1 sharded
    {3072, 3072, 3072, true,  132.47, true,  true,  2, 1, 1},  // L1 sharded
    {2048, 3072, 3072, false, 127.01, true,  true,  1, 1, 1},  // L1 sharded
    {2048, 3072, 3072, true,  128.65, true,  true,  1, 1, 1},  // L1 sharded
};

double compute_tflops(uint32_t M, uint32_t K, uint32_t N, double time_ns) {
    double flops = 2.0 * M * K * N;
    return flops / time_ns / 1e3;
}

int main() {
    bool use_eth = std::getenv("USE_ETH_DISPATCH") != nullptr;
    auto device = MeshDevice::create_unit_mesh(
        0, DEFAULT_L1_SMALL_SIZE, TRACE_REGION_SIZE, 1,
        use_eth ? DispatchCoreConfig{DispatchCoreType::ETH} : DispatchCoreConfig{}
    );

    auto grid = device->compute_with_storage_grid_size();
    int grid_x = grid.x, grid_y = grid.y;
    fmt::print("# Compute grid: {}x{} = {} cores\n", grid_x, grid_y, grid_x * grid_y);
    fmt::print("# ETH dispatch: {}\n", use_eth ? "true" : "false");
    fmt::print("# DataType: BFLOAT8_B, MathFidelity: HiFi2, packer_l1_acc: true\n");
    fmt::print("# Using reference benchmark program configs\n");

    MeshCommandQueue& cq = device->mesh_command_queue();

    // Create 8x8 core grid for sharding
    CoreRange core_range({0, 0}, {(size_t)(grid_x - 1), (size_t)(grid_y - 1)});
    CoreRangeSet core_grid({core_range});

    DeviceComputeKernelConfig compute_config = WormholeComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi2,
        .math_approx_mode = true,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = true,
        .throttle_level = ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE,
    };

    fmt::print("M,K,N,use_trace,in0_sharded,in0_block_w,out_block,subblock,time_ns,tflops,target_tflops,pct_of_target\n");

    for (const auto& cfg : TOP10_CONFIGS) {
        constexpr int tile_h = 32, tile_w = 32;

        // Calculate program config parameters per reference benchmark
        int in0_block_w = cfg.K / grid_x / 32 / cfg.in0_block_w_div;
        int per_core_M = cfg.M / grid_y / tile_h;
        int per_core_N = cfg.N / grid_x / tile_w;
        int out_block_h = per_core_M / cfg.num_out_blocks_h;
        int out_block_w = per_core_N / cfg.num_out_blocks_w;
        auto [out_subblock_h, out_subblock_w] = get_subblock_sizes(out_block_h, out_block_w, cfg.out_sharded);

        // Always provide explicit program config (reference benchmark does this for ALL shapes)
        MatmulProgramConfig program_config{
            .compute_with_storage_grid_size = {(size_t)grid_x, (size_t)grid_y},
            .in0_block_w = (size_t)in0_block_w,
            .out_subblock_h = (size_t)out_subblock_h,
            .out_subblock_w = (size_t)out_subblock_w,
            .out_block_h = (size_t)out_block_h,
            .out_block_w = (size_t)out_block_w,
            .per_core_M = (size_t)per_core_M,
            .per_core_N = (size_t)per_core_N,
            .transpose_mcast = false,
            .fused_activation = std::nullopt,
        };

        // Create tensors with appropriate memory config
        Tensor a, b;
        if (cfg.in0_sharded) {
            // L1 sharded memory config for in0
            auto in0_mem_config = ttnn::operations::data_movement::create_sharded_memory_config(
                Shape({1, 1, cfg.M, cfg.K}),
                core_grid,
                ShardStrategy::BLOCK,
                ShardOrientation::ROW_MAJOR
            );
            a = ones(Shape({1, 1, cfg.M, cfg.K}), DataType::BFLOAT8_B, TILE_LAYOUT, *device, in0_mem_config);
        } else {
            // DRAM memory config
            a = ones(Shape({1, 1, cfg.M, cfg.K}), DataType::BFLOAT8_B, TILE_LAYOUT, *device);
        }
        // in1 is always DRAM
        b = ones(Shape({1, 1, cfg.K, cfg.N}), DataType::BFLOAT8_B, TILE_LAYOUT, *device);

        // Output memory config
        std::optional<MemoryConfig> out_mem_config_opt;
        if (cfg.out_sharded) {
            out_mem_config_opt = MemoryConfig(TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1);
        }

        // Create output tile (32x32) to match reference benchmark
        tt::tt_metal::Tile output_tile({32, 32});

        auto do_matmul = [&]() {
            return ttnn::matmul(a, b,
                /*transpose_a=*/false,
                /*transpose_b=*/false,
                /*output_mem_config=*/out_mem_config_opt,
                /*output_dtype=*/DataType::BFLOAT8_B,
                /*program_config=*/program_config,
                /*activation=*/std::nullopt,
                /*compute_kernel_config=*/compute_config,
                /*core_grid=*/std::nullopt,
                /*output_tile=*/output_tile
            );
        };

        // Warmup
        for (int i = 0; i < N_WARMUP; i++) {
            auto warmup = do_matmul();
        }
        Finish(cq);

        double total_ns;
        if (cfg.use_trace) {
            auto trace_id = ttnn::operations::trace::begin_trace_capture(device.get(), std::nullopt);
            for (int i = 0; i < N_ITER; i++) {
                auto c = do_matmul();
            }
            ttnn::operations::trace::end_trace_capture(device.get(), trace_id, std::nullopt);

            auto start = std::chrono::high_resolution_clock::now();
            ttnn::operations::trace::execute_trace(device.get(), trace_id, std::nullopt, false);
            Finish(cq);
            auto end = std::chrono::high_resolution_clock::now();
            total_ns = std::chrono::duration<double, std::nano>(end - start).count();

            ttnn::operations::trace::release_trace(device.get(), trace_id);
        } else {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < N_ITER; i++) {
                auto c = do_matmul();
            }
            Finish(cq);
            auto end = std::chrono::high_resolution_clock::now();
            total_ns = std::chrono::duration<double, std::nano>(end - start).count();
        }

        double avg_ns = total_ns / N_ITER;
        double tflops = compute_tflops(cfg.M, cfg.K, cfg.N, avg_ns);
        double pct = (tflops / cfg.target_tflops) * 100.0;

        fmt::print("{},{},{},{},{},{},{}x{},{}x{},{:.0f},{:.2f},{:.2f},{:.1f}\n",
                   cfg.M, cfg.K, cfg.N, cfg.use_trace ? "true" : "false",
                   cfg.in0_sharded ? "L1" : "DRAM",
                   in0_block_w, out_block_h, out_block_w, out_subblock_h, out_subblock_w,
                   avg_ns, tflops, cfg.target_tflops, pct);
    }

    Finish(cq);
    ttnn::close_device(*device);
    return 0;
}
