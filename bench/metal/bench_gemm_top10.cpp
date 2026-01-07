// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GEMM Top 10 Benchmark - Step 1: BFLOAT4_B + ComputeKernelConfig
// Goal: Match reference TFLOPS from ~/tt-metal/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md
//
// Step 1: Use auto-selected program config, but with:
// - DataType::BFLOAT4_B
// - MathFidelity::LoFi
// - packer_l1_acc = true
// - ETH dispatch for 8x8 grid

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/trace.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <tt-metalium/distributed.hpp>

#include <chrono>
#include <vector>
#include <cstdlib>

using namespace ttnn;
using namespace tt::tt_metal::distributed;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;
using DispatchCoreType = tt::tt_metal::DispatchCoreType;

constexpr int N_WARMUP = 5;
constexpr int N_ITER = 100;
constexpr size_t TRACE_REGION_SIZE = 128 * 1024 * 1024;  // 128MB for trace

struct GemmConfig {
    uint32_t M, K, N;
    bool use_trace;
    double target_tflops;
};

// Top 10 configs from GEMM_FLOPS.md (Wormhole, BF4_B, LoFi)
std::vector<GemmConfig> TOP10_CONFIGS = {
    {4096,  4096,  4096,  true,  197.72},
    {16384, 16384, 16384, true,  200.09},
    {3072,  3072,  4096,  true,  196.73},
    {8192,  8192,  8192,  true,  196.15},
    {3072,  4096,  4096,  true,  195.60},
    {3072,  3072,  3072,  true,  194.27},
    {3072,  4096,  4096,  false, 190.38},
    {2048,  3072,  3072,  true,  190.16},
    {4096,  4096,  4096,  false, 193.70},
    {2048,  2048,  3072,  true,  184.37},
};

double compute_tflops(uint32_t M, uint32_t K, uint32_t N, double time_ns) {
    double flops = 2.0 * M * K * N;
    double seconds = time_ns / 1e9;
    return flops / seconds / 1e12;
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
    fmt::print("# DataType: BFLOAT4_B, MathFidelity: LoFi, packer_l1_acc: true\n");

    MeshCommandQueue& cq = device->mesh_command_queue();

    // WormholeComputeKernelConfig - key for performance
    DeviceComputeKernelConfig compute_config = WormholeComputeKernelConfig{
        .math_fidelity = MathFidelity::LoFi,
        .math_approx_mode = true,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = true,  // CRITICAL for performance
        .throttle_level = ttnn::operations::compute_throttle_utils::ThrottleLevel::NO_THROTTLE,
    };

    fmt::print("M,K,N,use_trace,time_ns,tflops,target_tflops,pct_of_target\n");

    for (const auto& cfg : TOP10_CONFIGS) {
        // Create tensors with BFLOAT4_B
        auto a = ones(Shape({cfg.M, cfg.K}), DataType::BFLOAT4_B, TILE_LAYOUT, *device);
        auto b = ones(Shape({cfg.K, cfg.N}), DataType::BFLOAT4_B, TILE_LAYOUT, *device);

        // Simple matmul call with compute config (auto-selected program config)
        // Using @ operator style - let TTNN choose optimal program config
        auto do_matmul = [&]() {
            return ttnn::matmul(a, b, false, false, std::nullopt,
                               DataType::BFLOAT4_B, std::nullopt,
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
        double pct = (tflops / cfg.target_tflops) * 100.0;

        fmt::print("{},{},{},{},{:.0f},{:.2f},{:.2f},{:.1f}\n",
                   cfg.M, cfg.K, cfg.N, cfg.use_trace ? "true" : "false",
                   avg_ns, tflops, cfg.target_tflops, pct);
    }

    Finish(cq);
    ttnn::close_device(*device);
    return 0;
}
