// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// BFP8 vs BF16 Matmul Benchmark - Optimized for Peak Performance
// Uses: LoFi for BFP8, L1 memory, trace API, optimal compute config

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/matmul/device/matmul_op.hpp>  // For program configs
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/uniform/uniform.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/trace.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>

#include <array>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace ttnn;
using namespace tt::tt_metal::distributed;

constexpr int N_WARMUP = 10;
constexpr int N_ITER = 50;

// Wormhole peak TFLOP/s from tech report (with full optimization)
constexpr double PEAK_BFP8_TFLOPS = 300.0;  // LoFi, L1, traced
constexpr double PEAK_BF16_TFLOPS = 190.0;  // HiFi4, L1, traced

struct TimingResult {
    double mean_us;
    double std_us;
};

struct BenchResult {
    uint32_t size;
    double bfp8_mean_us, bfp8_std_us;
    double bf16_mean_us, bf16_std_us;
    double speedup;
    double max_error, mean_error;
    double bfp8_tflops, bf16_tflops;
};

// Create optimal compute kernel config for Wormhole
DeviceComputeKernelConfig create_compute_config(DataType dtype) {
    // LoFi for BFP8 (2x throughput vs HiFi2!), HiFi4 for BF16
    MathFidelity fidelity = (dtype == DataType::BFLOAT8_B)
        ? MathFidelity::LoFi    // CRITICAL: 2x throughput vs HiFi2
        : MathFidelity::HiFi4;

    return WormholeComputeKernelConfig{
        .math_fidelity = fidelity,
        .math_approx_mode = true,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = true,  // CRITICAL for peak performance
        .dst_full_sync_en = false
    };
}

// Calculate TFLOP/s: 2 * N^3 / time_us / 1e6
double calc_tflops(uint32_t size, double time_us) {
    double flops = 2.0 * size * size * size;
    return flops / time_us / 1e6;
}

// Create optimal program config matching tech report benchmark
// Based on test_benchmark.py from tt-metal/tests/ttnn/unit_tests/benchmarks/
// Returns nullopt if size doesn't divide evenly by grid or would exceed L1
std::optional<ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>
create_program_config(uint32_t size, uint32_t grid_x, uint32_t grid_y) {
    constexpr uint32_t TILE_SIZE = 32;
    constexpr uint32_t MAX_L1_SIZE = 1400000;  // ~1.4MB safe limit
    constexpr uint32_t BYTES_PER_TILE_BF16 = 32 * 32 * 2;  // 2KB

    // Check if size divides evenly by grid
    if (size % (grid_y * TILE_SIZE) != 0 || size % (grid_x * TILE_SIZE) != 0) {
        return std::nullopt;  // Let TTNN auto-select
    }

    // Calculate per-core tile counts
    uint32_t per_core_M = size / grid_y / TILE_SIZE;
    uint32_t per_core_N = size / grid_x / TILE_SIZE;
    uint32_t in0_block_w = size / grid_x / TILE_SIZE;  // K blocking

    // Estimate L1 usage: input tiles + output tiles (rough estimate)
    // Each core needs: in0 tiles + in1 tiles + out tiles
    uint32_t tiles_per_core = per_core_M * in0_block_w + in0_block_w * per_core_N + per_core_M * per_core_N;
    uint32_t estimated_l1 = tiles_per_core * BYTES_PER_TILE_BF16 * 2;  // 2x for double buffering

    if (estimated_l1 > MAX_L1_SIZE) {
        return std::nullopt;  // Let TTNN auto-select with its own blocking
    }

    // Subblock selection - common good choices from benchmark
    // out_subblock_h * out_subblock_w should be 8 for best performance
    uint32_t out_subblock_h = 4;
    uint32_t out_subblock_w = 2;

    // Adjust if tiles per core don't divide evenly
    while (per_core_M % out_subblock_h != 0 && out_subblock_h > 1) out_subblock_h--;
    while (per_core_N % out_subblock_w != 0 && out_subblock_w > 1) out_subblock_w--;

    return ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
        .compute_with_storage_grid_size = {grid_x, grid_y},
        .in0_block_w = in0_block_w,
        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
        .out_block_h = per_core_M,
        .out_block_w = per_core_N,
        .per_core_M = per_core_M,
        .per_core_N = per_core_N,
        .transpose_mcast = false,
    };
}

// Benchmark matmul with specific data type, optimal config, and trace API
TimingResult benchmark_matmul_dtype(MeshDevice& device, uint32_t size, DataType dtype) {
    std::array<uint32_t, 2> shape = {size, size};

    // Use DRAM for consistency with tech report out-of-box config
    auto mem_config = DRAM_MEMORY_CONFIG;

    auto a = ones(Shape(shape), dtype, TILE_LAYOUT, device);
    auto b = ones(Shape(shape), dtype, TILE_LAYOUT, device);

    MeshCommandQueue& cq = device.mesh_command_queue();
    Finish(cq);

    // Get optimal compute config
    auto compute_config = create_compute_config(dtype);

    // Get actual compute grid size from device (avoid dispatch cores)
    auto compute_grid = device.compute_with_storage_grid_size();
    uint32_t grid_x = compute_grid.x;
    uint32_t grid_y = compute_grid.y;

    // Get optimal program config (matching tech report benchmark)
    auto program_config = create_program_config(size, grid_x, grid_y);

    // Warmup (compile kernels)
    for (int i = 0; i < N_WARMUP; i++) {
        auto warmup = ttnn::matmul(
            a, b,
            /*transpose_a=*/false,
            /*transpose_b=*/false,
            /*memory_config=*/mem_config,
            /*dtype=*/std::nullopt,
            /*program_config=*/program_config,
            /*activation=*/std::nullopt,
            /*compute_kernel_config=*/compute_config,
            /*core_grid=*/std::nullopt  // Not needed with program_config
        );
        Finish(cq);
    }

    // Capture trace for minimal host overhead
    auto trace_id = operations::trace::begin_trace_capture(&device, std::nullopt);
    auto c_traced = ttnn::matmul(
        a, b,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        /*memory_config=*/mem_config,
        /*dtype=*/std::nullopt,
        /*program_config=*/program_config,
        /*activation=*/std::nullopt,
        /*compute_kernel_config=*/compute_config,
        /*core_grid=*/std::nullopt
    );
    operations::trace::end_trace_capture(&device, trace_id, std::nullopt);

    // Timed trace replays
    std::vector<double> times_us(N_ITER);

    for (int i = 0; i < N_ITER; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        operations::trace::execute_trace(&device, trace_id, std::nullopt, false);
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_us[i] = std::chrono::duration<double, std::micro>(end - start).count();
    }

    // Release trace
    operations::trace::release_trace(&device, trace_id);

    // Stats
    double sum = std::accumulate(times_us.begin(), times_us.end(), 0.0);
    double mean = sum / N_ITER;

    double sq_sum = 0.0;
    for (double t : times_us) {
        sq_sum += (t - mean) * (t - mean);
    }
    double std_dev = std::sqrt(sq_sum / N_ITER);

    return TimingResult{mean, std_dev};
}

// Compute error between BFP8 and BF16 matmul outputs
std::pair<double, double> compute_error(MeshDevice& device, uint32_t size) {
    std::array<uint32_t, 2> shape = {size, size};
    MeshCommandQueue& cq = device.mesh_command_queue();

    // Create random tensors in BF16
    auto a_bf16 = ones(Shape(shape), DataType::BFLOAT16, TILE_LAYOUT, device);
    auto b_bf16 = ones(Shape(shape), DataType::BFLOAT16, TILE_LAYOUT, device);
    a_bf16 = ttnn::uniform(a_bf16, -1.0f, 1.0f, 42, std::nullopt, std::nullopt);
    b_bf16 = ttnn::uniform(b_bf16, -1.0f, 1.0f, 43, std::nullopt, std::nullopt);
    Finish(cq);

    // Convert to BFP8
    auto a_bfp8 = ttnn::typecast(a_bf16, DataType::BFLOAT8_B);
    auto b_bfp8 = ttnn::typecast(b_bf16, DataType::BFLOAT8_B);
    Finish(cq);

    // Matmul in both dtypes with optimal configs
    auto bfp8_config = create_compute_config(DataType::BFLOAT8_B);
    auto bf16_config = create_compute_config(DataType::BFLOAT16);

    // Get actual compute grid size from device
    auto compute_grid = device.compute_with_storage_grid_size();
    auto program_config = create_program_config(size, compute_grid.x, compute_grid.y);

    auto c_bfp8 = ttnn::matmul(a_bfp8, b_bfp8, false, false, std::nullopt, std::nullopt,
                               program_config, std::nullopt, bfp8_config, std::nullopt);
    auto c_bf16 = ttnn::matmul(a_bf16, b_bf16, false, false, std::nullopt, std::nullopt,
                               program_config, std::nullopt, bf16_config, std::nullopt);
    Finish(cq);

    // Convert BFP8 result to BF16 for comparison
    auto c_bfp8_bf16 = ttnn::typecast(c_bfp8, DataType::BFLOAT16);
    Finish(cq);

    // Move to CPU
    auto c_bfp8_cpu = c_bfp8_bf16.cpu(true);
    auto c_bf16_cpu = c_bf16.cpu(true);

    // Extract values
    auto bfp8_vals = c_bfp8_cpu.to_vector<bfloat16>();
    auto bf16_vals = c_bf16_cpu.to_vector<bfloat16>();

    // Compute error statistics
    double max_err = 0.0;
    double sum_err = 0.0;
    for (size_t i = 0; i < bfp8_vals.size(); i++) {
        double diff = std::abs(static_cast<float>(bfp8_vals[i]) - static_cast<float>(bf16_vals[i]));
        max_err = std::max(max_err, diff);
        sum_err += diff;
    }
    double mean_err = sum_err / bfp8_vals.size();

    return {max_err, mean_err};
}

int main() {
    // Use ETH dispatch for full 8x8 grid on N300 cluster
    // Pattern from: tests/ttnn/unit_tests/gtests/ttnn_test_fixtures.hpp:82-87
    using tt::tt_metal::DispatchCoreConfig;
    using tt::tt_metal::DispatchCoreType;
    auto device = MeshDevice::create_unit_mesh(
        0,                           // device_id
        DEFAULT_L1_SMALL_SIZE,       // l1_small_size
        128 * 1024 * 1024,           // trace_region_size
        1,                           // num_command_queues
        DispatchCoreConfig{DispatchCoreType::ETH}
    );

    // Print actual grid size
    auto grid = device->compute_with_storage_grid_size();
    fmt::print("# Compute grid: {}x{} = {} cores\n", grid.x, grid.y, grid.x * grid.y);

    // Sizes must divide evenly by grid_x*32 and grid_y*32
    // For 8x8 grid: 8*32 = 256, so powers of 2 work perfectly
    // Note: 4096+ sizes may exceed L1 with 8x8 grid - use nullopt program_config
    std::vector<uint32_t> sizes = {512, 1024, 2048, 4096, 8192};
    std::vector<BenchResult> results;

    // CSV header with TFLOP/s columns
    fmt::print("size,bfp8_mean_us,bfp8_std_us,bf16_mean_us,bf16_std_us,speedup,bfp8_tflops,bf16_tflops,bfp8_util,bf16_util,max_error,mean_error\n");

    for (uint32_t size : sizes) {
        // Benchmark BFP8
        auto bfp8_timing = benchmark_matmul_dtype(*device, size, DataType::BFLOAT8_B);

        // Benchmark BF16
        auto bf16_timing = benchmark_matmul_dtype(*device, size, DataType::BFLOAT16);

        // Compute error
        auto [max_error, mean_error] = compute_error(*device, size);

        // Calculate metrics
        double speedup = bf16_timing.mean_us / bfp8_timing.mean_us;
        double bfp8_tflops = calc_tflops(size, bfp8_timing.mean_us);
        double bf16_tflops = calc_tflops(size, bf16_timing.mean_us);
        double bfp8_util = bfp8_tflops / PEAK_BFP8_TFLOPS * 100;
        double bf16_util = bf16_tflops / PEAK_BF16_TFLOPS * 100;

        // Print CSV row
        fmt::print("{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.1f},{:.1f},{:.6f},{:.6f}\n",
                   size, bfp8_timing.mean_us, bfp8_timing.std_us,
                   bf16_timing.mean_us, bf16_timing.std_us,
                   speedup, bfp8_tflops, bf16_tflops,
                   bfp8_util, bf16_util,
                   max_error, mean_error);
    }

    // Cleanup
    Finish(device->mesh_command_queue());
    ttnn::close_device(*device);

    return 0;
}
