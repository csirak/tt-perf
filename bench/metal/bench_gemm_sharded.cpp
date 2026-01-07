// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GEMM Sharded Benchmark - L1 Sharding for smaller matrices
// Goal: Match reference TFLOPS for configs that need L1 sharding

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/trace.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/data_movement/common/common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>

#include <chrono>
#include <vector>
#include <cstdlib>

using namespace ttnn;
using namespace tt::tt_metal::distributed;
using namespace tt::tt_metal;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;
using DispatchCoreType = tt::tt_metal::DispatchCoreType;
using MatmulProgramConfig = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig;
using ShardStrategy = ttnn::operations::data_movement::ShardStrategy;

constexpr int N_WARMUP = 5;
constexpr int N_ITER = 100;
constexpr size_t TRACE_REGION_SIZE = 128 * 1024 * 1024;

struct GemmConfig {
    uint32_t M, K, N;
    bool use_trace;
    double target_tflops;
};

// Configs that need L1 sharding
std::vector<GemmConfig> SHARDED_CONFIGS = {
    {3072, 3072, 4096, true, 196.73},
    {3072, 3072, 3072, true, 194.27},
    {2048, 3072, 3072, true, 190.16},
    {2048, 2048, 3072, true, 184.37},
};

double compute_tflops(uint32_t M, uint32_t K, uint32_t N, double time_ns) {
    double flops = 2.0 * M * K * N;
    double seconds = time_ns / 1e9;
    return flops / seconds / 1e12;
}

// Subblock selection from test_benchmark.py
std::pair<int, int> get_subblock_sizes(int m_tiles, int n_tiles, bool out_sharded) {
    std::vector<std::pair<int, int>> choices = {
        {4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2}, {2, 3},
        {6, 1}, {1, 6}, {5, 1}, {1, 5}, {2, 2}, {4, 1}, {1, 4}, {3, 1},
        {1, 3}, {2, 1}, {1, 2}, {1, 1}
    };
    for (auto [h, w] : choices) {
        if (out_sharded && (n_tiles % w != 0 || h != 1)) continue;
        if (m_tiles % h == 0 && n_tiles % w == 0) return {h, w};
    }
    return {1, 1};
}

int main() {
    // Open device with ETH dispatch for 8x8 grid
    bool use_eth = std::getenv("USE_ETH_DISPATCH") != nullptr;
    auto device = MeshDevice::create_unit_mesh(
        0, DEFAULT_L1_SMALL_SIZE, TRACE_REGION_SIZE, 1,
        use_eth ? DispatchCoreConfig{DispatchCoreType::ETH} : DispatchCoreConfig{}
    );

    auto grid = device->compute_with_storage_grid_size();
    int grid_x = grid.x, grid_y = grid.y;
    fmt::print("# Compute grid: {}x{} = {} cores\n", grid_x, grid_y, grid_x * grid_y);
    fmt::print("# ETH dispatch: {}\n", use_eth ? "true" : "false");
    fmt::print("# L1 Sharded: in0=L1, in1=DRAM, out=L1\n");
    fmt::print("# DataType: BFLOAT4_B, MathFidelity: LoFi, packer_l1_acc: true\n");

    MeshCommandQueue& cq = device->mesh_command_queue();

    // WormholeComputeKernelConfig
    DeviceComputeKernelConfig compute_config = WormholeComputeKernelConfig{
        .math_fidelity = MathFidelity::LoFi,
        .math_approx_mode = true,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = true,
        .throttle_level = ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE,
    };

    // Create CoreRangeSet for 8x8 grid
    CoreRange core_range({0, 0}, {(size_t)grid_x - 1, (size_t)grid_y - 1});
    CoreRangeSet core_grid({core_range});

    // L1 sharded output memory config
    MemoryConfig out_mem_config(TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1);

    fmt::print("M,K,N,use_trace,time_ns,tflops,target_tflops,pct_of_target\n");

    for (const auto& cfg : SHARDED_CONFIGS) {
        // Calculate per-core dimensions (tiles per core)
        int per_core_M = cfg.M / grid_y / 32;
        int per_core_N = cfg.N / grid_x / 32;
        int in0_block_w = cfg.K / grid_x / 32;

        // Block sizes
        int out_block_h = per_core_M;
        int out_block_w = per_core_N;
        auto [out_subblock_h, out_subblock_w] = get_subblock_sizes(out_block_h, out_block_w, true);

        // L1 sharded memory config for in0
        auto in0_mem_config = ttnn::operations::data_movement::create_sharded_memory_config(
            Shape({1, 1, cfg.M, cfg.K}),
            core_grid,
            ShardStrategy::BLOCK,
            ShardOrientation::ROW_MAJOR
        );

        // Create tensors - in0 sharded on L1, in1 on DRAM
        auto a = ones(Shape({1, 1, cfg.M, cfg.K}), DataType::BFLOAT4_B, TILE_LAYOUT, *device, in0_mem_config);
        auto b = ones(Shape({1, 1, cfg.K, cfg.N}), DataType::BFLOAT4_B, TILE_LAYOUT, *device);

        // MatmulMultiCoreReuseMultiCastProgramConfig
        ttnn::operations::matmul::MatmulProgramConfig program_config = MatmulProgramConfig{
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

        auto do_matmul = [&]() {
            return ttnn::matmul(a, b, false, false, out_mem_config,
                               DataType::BFLOAT4_B, program_config,
                               std::nullopt, compute_config);
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

        fmt::print("{},{},{},{},{:.0f},{:.2f},{:.2f},{:.1f}\n",
                   cfg.M, cfg.K, cfg.N, cfg.use_trace ? "true" : "false",
                   avg_ns, tflops, cfg.target_tflops, pct);
    }

    Finish(cq);
    ttnn::close_device(*device);
    return 0;
}
