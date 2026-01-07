// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GEMM BF16 Top 10 Benchmark
// Targets top-performing BF16 matmul shapes from gemm_flops_n150_tuned.csv
//
// Key settings:
// - DataType: BFLOAT16
// - MathFidelity: HiFi2 (1.5x faster than HiFi4)
// - L1 sharding for shapes ≤3072 (asymmetric: in0=L1, in1=DRAM, out=L1)
// - DRAM for shapes >3072
// - ETH dispatch for 8x8 grid
// - Trace API for all shapes
// - packer_l1_acc = true

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/matmul/device/matmul_op.hpp>  // For MatmulProgramConfig
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
using ShardStrategy = ttnn::operations::data_movement::ShardStrategy;

constexpr int N_WARMUP = 5;
constexpr int N_ITER = 100;
constexpr size_t TRACE_REGION_SIZE = 128 * 1024 * 1024;  // 128MB for trace

struct GemmConfig {
    uint32_t M, K, N;
    bool use_trace;
    bool use_l1;  // L1 sharding for ≤3072, DRAM for larger
    double target_tflops;  // 0.0 = no target (discovery mode)
    // Program config tuning parameters (0 = auto-select)
    uint32_t in0_block_w_div;     // divisor for in0_block_w
    uint32_t num_out_blocks_h;    // divisor for out_block_h
    uint32_t num_out_blocks_w;    // divisor for out_block_w
    const char* category;         // shape category (perf, gpt2-small, gpt2-mini)
};

// Top-N BF16 HiFi2 configs: high-performance shapes + GPT-2 practical shapes
// Format: {M, K, N, use_trace, use_l1, target_tflops, in0_block_w_div, num_out_blocks_h, num_out_blocks_w, category}
std::vector<GemmConfig> BF16_TOPN = {
    // === High-Performance Shapes (from tuned benchmark) ===
    // Small shapes (≤1024) - L1 sharded
    {512,  1024, 1024, true, true,  36.20, 0, 0, 0, "perf"},
    {512,  1024, 2048, true, true,  41.27, 0, 0, 0, "perf"},
    {1024, 1024, 1024, true, true,  56.20, 0, 0, 0, "perf"},
    {1024, 1024, 2048, true, true,  63.87, 0, 0, 0, "perf"},
    {1024, 2048, 2048, true, true,  69.10, 0, 0, 0, "perf"},

    // Medium shapes - 2048³ L1, others DRAM with tuning
    {2048, 2048, 2048, true, true,  76.64, 0, 0, 0, "perf"},
    {2048, 2048, 3072, true, false, 77.06, 1, 1, 1, "perf"},
    {2048, 3072, 3072, true, false, 76.61, 2, 1, 1, "perf"},

    // Large shapes - DRAM with explicit program config
    {8192,  8192,  8192,  true, false, 81.26, 2, 4, 4, "perf"},
    {16384, 16384, 16384, true, false, 85.51, 4, 8, 8, "perf"},

    // === GPT-2 Small (768 dim, batch=32, seq=256 → M=8192) ===
    {8192, 768,  768,  true, false, 0.0, 0, 0, 0, "gpt2-small"},  // QKV/Output projection
    {8192, 768,  3072, true, false, 0.0, 0, 0, 0, "gpt2-small"},  // FFN up
    {8192, 3072, 768,  true, false, 0.0, 0, 0, 0, "gpt2-small"},  // FFN down

    // === Mini GPT (512 dim, batch=32, seq=256 → M=8192) ===
    {8192, 512,  512,  true, false, 0.0, 0, 0, 0, "gpt2-mini"},   // QKV/Output projection
    {8192, 512,  2048, true, false, 0.0, 0, 0, 0, "gpt2-mini"},   // FFN up
    {8192, 2048, 512,  true, false, 0.0, 0, 0, 0, "gpt2-mini"},   // FFN down
};

double compute_tflops(uint32_t M, uint32_t K, uint32_t N, double time_ns) {
    double flops = 2.0 * M * K * N;
    double seconds = time_ns / 1e9;
    return flops / seconds / 1e12;
}

int main() {
    // Open device with ETH dispatch for 8x8 grid (default ON unless USE_ETH_DISPATCH=0)
    const char* eth_env = std::getenv("USE_ETH_DISPATCH");
    bool use_eth = (eth_env == nullptr) || (std::string(eth_env) != "0");

    // Use 2 command queues for better async overlap (matches Python benchmark)
    constexpr int NUM_CQS = 2;
    auto device = MeshDevice::create_unit_mesh(
        0, DEFAULT_L1_SMALL_SIZE, TRACE_REGION_SIZE, NUM_CQS,
        use_eth ? DispatchCoreConfig{DispatchCoreType::ETH} : DispatchCoreConfig{}
    );

    auto grid = device->compute_with_storage_grid_size();
    int grid_x = grid.x, grid_y = grid.y;
    fmt::print("# Compute grid: {}x{} = {} cores\n", grid_x, grid_y, grid_x * grid_y);
    fmt::print("# ETH dispatch: {}, CQs: {}\n", use_eth ? "true" : "false", NUM_CQS);
    fmt::print("# DataType: BFLOAT16, MathFidelity: HiFi2, packer_l1_acc: true\n");

    MeshCommandQueue& cq = device->mesh_command_queue();

    // WormholeComputeKernelConfig - HiFi2 for BF16
    DeviceComputeKernelConfig compute_config = WormholeComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi2,
        .math_approx_mode = true,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = true,  // CRITICAL for performance
        .throttle_level = ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE,
    };

    // Create CoreRangeSet for 8x8 grid (used for L1 sharding)
    CoreRange core_range({0, 0}, {(size_t)grid_x - 1, (size_t)grid_y - 1});
    CoreRangeSet core_grid({core_range});

    // L1 sharded output memory config
    MemoryConfig l1_out_config(TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1);

    fmt::print("M,K,N,category,use_trace,sharding,time_ns,tflops,target_tflops,pct_of_target\n");

    for (const auto& cfg : BF16_TOPN) {
        ttnn::Tensor a, b;
        std::optional<MemoryConfig> out_mem_config = std::nullopt;

        // Check if shape can be evenly sharded on 8x8 grid
        // M must divide by (grid_y * 32), N must divide by (grid_x * 32)
        bool can_shard = cfg.use_l1 &&
                         (cfg.M % (grid_y * 32) == 0) &&
                         (cfg.N % (grid_x * 32) == 0) &&
                         (cfg.K % (grid_x * 32) == 0);

        std::string sharding_type = "DRAM";

        if (can_shard) {
            // L1 sharded: in0=L1 (block sharded), in1=DRAM, out=L1
            // Use 4D shape for sharding (1, 1, M, K)
            auto in0_mem_config = ttnn::operations::data_movement::create_sharded_memory_config(
                Shape({1, 1, cfg.M, cfg.K}),
                core_grid,
                ShardStrategy::BLOCK,
                ShardOrientation::ROW_MAJOR
            );
            a = ones(Shape({1, 1, cfg.M, cfg.K}), DataType::BFLOAT16, TILE_LAYOUT, *device, in0_mem_config);
            b = ones(Shape({1, 1, cfg.K, cfg.N}), DataType::BFLOAT16, TILE_LAYOUT, *device);
            out_mem_config = l1_out_config;
            sharding_type = "L1";
        } else {
            // DRAM: standard 2D tensors, no sharding
            a = ones(Shape({cfg.M, cfg.K}), DataType::BFLOAT16, TILE_LAYOUT, *device);
            b = ones(Shape({cfg.K, cfg.N}), DataType::BFLOAT16, TILE_LAYOUT, *device);
        }

        // Create program config - explicit for large DRAM shapes, auto for others
        std::optional<ttnn::operations::matmul::MatmulProgramConfig> program_config = std::nullopt;

        if (cfg.in0_block_w_div > 0 && !can_shard) {
            // Explicit program config for large DRAM shapes
            uint32_t per_core_M = cfg.M / grid_y / 32;  // in tiles
            uint32_t per_core_N = cfg.N / grid_x / 32;
            uint32_t in0_block_w = cfg.K / grid_x / 32 / cfg.in0_block_w_div;
            uint32_t out_block_h = per_core_M / cfg.num_out_blocks_h;
            uint32_t out_block_w = per_core_N / cfg.num_out_blocks_w;

            // Get optimal subblock sizes (max 8, must divide evenly)
            auto get_subblock = [](uint32_t block_h, uint32_t block_w) -> std::pair<uint32_t, uint32_t> {
                // Try subblock sizes that give subblock_hw <= 8 (or 4 for fp32 acc)
                std::vector<std::pair<uint32_t,uint32_t>> choices = {
                    {4,2}, {2,4}, {8,1}, {1,8}, {7,1}, {1,7},
                    {3,2}, {2,3}, {6,1}, {1,6}, {5,1}, {1,5},
                    {2,2}, {4,1}, {1,4}, {3,1}, {1,3}, {2,1}, {1,2}, {1,1}
                };
                for (auto [sh, sw] : choices) {
                    if (block_h % sh == 0 && block_w % sw == 0) {
                        return {sh, sw};
                    }
                }
                return {1, 1};
            };
            auto [out_subblock_h, out_subblock_w] = get_subblock(out_block_h, out_block_w);

            program_config = ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
                .compute_with_storage_grid_size = CoreCoord{(size_t)grid_x, (size_t)grid_y},
                .in0_block_w = in0_block_w,
                .out_subblock_h = out_subblock_h,
                .out_subblock_w = out_subblock_w,
                .out_block_h = out_block_h,
                .out_block_w = out_block_w,
                .per_core_M = per_core_M,
                .per_core_N = per_core_N,
                .transpose_mcast = false,
                .fused_activation = std::nullopt,
                .fuse_batch = true,
            };
        }

        // Matmul with compute config
        auto do_matmul = [&]() {
            return ttnn::matmul(a, b, false, false, out_mem_config,
                               DataType::BFLOAT16, program_config,
                               std::nullopt, compute_config);
        };

        // Warmup
        for (int i = 0; i < N_WARMUP; i++) {
            auto warmup = do_matmul();
        }
        Finish(cq);

        double total_ns;
        if (cfg.use_trace) {
            // Trace capture
            auto trace_id = ttnn::operations::trace::begin_trace_capture(device.get(), std::nullopt);
            for (int i = 0; i < N_ITER; i++) {
                auto c = do_matmul();
            }
            ttnn::operations::trace::end_trace_capture(device.get(), trace_id, std::nullopt);

            // Timed trace execution
            auto start = std::chrono::high_resolution_clock::now();
            ttnn::operations::trace::execute_trace(device.get(), trace_id, std::nullopt, false);
            Finish(cq);
            auto end = std::chrono::high_resolution_clock::now();
            total_ns = std::chrono::duration<double, std::nano>(end - start).count();

            ttnn::operations::trace::release_trace(device.get(), trace_id);
        } else {
            // Non-traced timed run
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

        // Handle target=0.0 (discovery mode) - show "N/A" for percentage
        if (cfg.target_tflops > 0.0) {
            double pct = (tflops / cfg.target_tflops) * 100.0;
            fmt::print("{},{},{},{},{},{},{:.0f},{:.2f},{:.2f},{:.1f}\n",
                       cfg.M, cfg.K, cfg.N, cfg.category,
                       cfg.use_trace ? "true" : "false",
                       sharding_type,
                       avg_ns, tflops, cfg.target_tflops, pct);
        } else {
            fmt::print("{},{},{},{},{},{},{:.0f},{:.2f},{:.2f},N/A\n",
                       cfg.M, cfg.K, cfg.N, cfg.category,
                       cfg.use_trace ? "true" : "false",
                       sharding_type,
                       avg_ns, tflops, cfg.target_tflops);
        }
    }

    Finish(cq);
    ttnn::close_device(*device);
    return 0;
}
