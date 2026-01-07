// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: Parallel MLPs with Core Grid Partitioning
// Tests data parallel throughput using the core_grid parameter to restrict
// each MLP to a subset of cores.
//
// Approach:
// - Split the 8x7 core grid into N vertical stripes
// - Each MLP uses a different stripe via core_grid parameter
// - All MLPs share the same weights (loaded once)
// - Measures if independent core regions enable parallel execution
//
// Compares:
// - Solo: Full device (56 cores), 1 MLP
// - 2-way: 2 stripes (4x7 = 28 cores each), 2 MLPs
// - 4-way: 4 stripes (2x7 = 14 cores each), 4 MLPs

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/memory_config/memory_config.hpp>
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
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

constexpr int N_WARMUP = 5;
constexpr int N_TIMED = 20;

struct SizeConfig {
    const char* name;
    uint32_t batch;
    uint32_t dim;
};

// MLP activations for one partition
struct MLPActivations {
    Tensor x;
    Tensor dout;
};

// Shared weights
struct MLPWeights {
    Tensor w1, b1, w2, b2;
};

// Create weights for MLP (shared across all partitions)
MLPWeights create_weights(MeshDevice& dev, uint32_t dim) {
    return {
        ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        zeros(Shape({1, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        zeros(Shape({1, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
    };
}

// Create activations for one partition
MLPActivations create_activations(MeshDevice& dev, uint32_t batch, uint32_t dim) {
    return {
        ones(Shape({batch, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        ones(Shape({batch, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
    };
}

// Run forward+backward for one MLP (no core_grid - uses full device)
void run_mlp(const MLPActivations& act, const MLPWeights& wt) {
    // Forward
    auto mm1 = ttnn::matmul(act.x, wt.w1, false, true);
    auto h_pre = ttnn::add(mm1, wt.b1);
    auto h = ttnn::relu(h_pre);
    auto mm2 = ttnn::matmul(h, wt.w2, false, true);
    auto out = ttnn::add(mm2, wt.b2);

    // Backward
    auto dh = ttnn::matmul(act.dout, wt.w2, false, false);
    auto relu_mask = ttnn::gtz(h_pre);
    auto dh_pre = ttnn::multiply(dh, relu_mask);
    auto dx = ttnn::matmul(dh_pre, wt.w1, false, false);
}

// Run forward+backward with explicit core_grid
void run_mlp_partitioned(const MLPActivations& act, const MLPWeights& wt,
                         const CoreGrid& grid) {
    // Forward - use core_grid to restrict matmul to specific cores
    auto mm1 = ttnn::matmul(act.x, wt.w1, false, true, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, std::nullopt, grid);
    auto h_pre = ttnn::add(mm1, wt.b1);
    auto h = ttnn::relu(h_pre);
    auto mm2 = ttnn::matmul(h, wt.w2, false, true, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, std::nullopt, grid);
    auto out = ttnn::add(mm2, wt.b2);

    // Backward
    auto dh = ttnn::matmul(act.dout, wt.w2, false, false, std::nullopt, std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, grid);
    auto relu_mask = ttnn::gtz(h_pre);
    auto dh_pre = ttnn::multiply(dh, relu_mask);
    auto dx = ttnn::matmul(dh_pre, wt.w1, false, false, std::nullopt, std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, grid);
}

// Benchmark: Solo MLP on full device (baseline)
double bench_solo(MeshDevice& dev, const SizeConfig& size) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    auto weights = create_weights(dev, size.dim);
    auto acts = create_activations(dev, size.batch, size.dim);
    Finish(cq);

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        run_mlp(acts, weights);
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        run_mlp(acts, weights);
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

// Benchmark: N parallel MLPs without partitioning (naive - same cores for all)
double bench_naive_parallel(MeshDevice& dev, const SizeConfig& size, int n_parallel) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    auto weights = create_weights(dev, size.dim);
    std::vector<MLPActivations> acts_list;
    for (int i = 0; i < n_parallel; i++) {
        acts_list.push_back(create_activations(dev, size.batch, size.dim));
    }
    Finish(cq);

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        for (const auto& acts : acts_list) {
            run_mlp(acts, weights);
        }
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (const auto& acts : acts_list) {
            run_mlp(acts, weights);
        }
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

// Benchmark: N parallel MLPs with core_grid partitioning
// Each MLP gets a vertical stripe of the core grid
double bench_partitioned_parallel(MeshDevice& dev, const SizeConfig& size, int n_parallel) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    auto grid = dev.compute_with_storage_grid_size();
    uint32_t cols_per_partition = grid.x / n_parallel;

    // Create core grids for each partition (vertical stripes)
    std::vector<CoreGrid> partition_grids;
    for (int i = 0; i < n_parallel; i++) {
        // CoreGrid uses (x, y) where x=width, y=height
        partition_grids.emplace_back(cols_per_partition, grid.y);
    }

    auto weights = create_weights(dev, size.dim);
    std::vector<MLPActivations> acts_list;
    for (int i = 0; i < n_parallel; i++) {
        acts_list.push_back(create_activations(dev, size.batch, size.dim));
    }
    Finish(cq);

    // Warmup - each MLP uses its own core region
    for (int w = 0; w < N_WARMUP; w++) {
        for (int i = 0; i < n_parallel; i++) {
            run_mlp_partitioned(acts_list[i], weights, partition_grids[i]);
        }
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        // Enqueue all MLPs with their respective core grids
        for (int i = 0; i < n_parallel; i++) {
            run_mlp_partitioned(acts_list[i], weights, partition_grids[i]);
        }
        Finish(cq);

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

int main() {
    auto device = MeshDevice::create_unit_mesh(0);

    auto grid_size = device->compute_with_storage_grid_size();
    fmt::print("# Device core grid: {}x{} = {} cores\n", grid_size.x, grid_size.y,
               grid_size.x * grid_size.y);

    std::vector<SizeConfig> sizes = {
        {"small",  32,   256},
        {"medium", 256,  512},
        {"large",  512,  1024},
    };

    // Parallel counts must evenly divide grid.x (8)
    std::vector<int> parallel_counts = {1, 2, 4};

    fmt::print("# Parallel MLP with Core Grid Partitioning\n");
    fmt::print("# Naive: All MLPs use full core grid (sequential, overlapping resources)\n");
    fmt::print("# Partitioned: Each MLP restricted to dedicated core stripe via core_grid\n");
    fmt::print("# Warmup: {}, Timed iterations: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");
    fmt::print("# Congestion ratio = total_throughput / (N × solo_throughput)\n");
    fmt::print("# Ratio = 1.0 means perfect scaling, <1.0 means congestion\n");
    fmt::print("#\n");
    fmt::print("size,batch,dim,n_parallel,cores_per_partition,naive_ms,partitioned_ms,solo_ms,");
    fmt::print("naive_throughput,partitioned_throughput,naive_congestion,partitioned_congestion,speedup\n");

    for (const auto& size : sizes) {
        // Baseline: solo MLP on full device
        double solo_ms = bench_solo(*device, size);
        double solo_throughput = size.batch / solo_ms * 1000.0;

        for (int n : parallel_counts) {
            uint32_t cols_per_partition = grid_size.x / n;
            uint32_t cores_per_partition = cols_per_partition * grid_size.y;

            double naive_ms = bench_naive_parallel(*device, size, n);
            double partitioned_ms = (n == 1) ? naive_ms : bench_partitioned_parallel(*device, size, n);

            // Calculate throughputs
            double naive_throughput = (n * size.batch) / naive_ms * 1000.0;
            double partitioned_throughput = (n * size.batch) / partitioned_ms * 1000.0;

            // Congestion ratios
            double naive_congestion = naive_throughput / (n * solo_throughput);
            double partitioned_congestion = partitioned_throughput / (n * solo_throughput);

            // Speedup of partitioned over naive
            double speedup = naive_ms / partitioned_ms;

            fmt::print("{},{},{},{},{},{:.3f},{:.3f},{:.3f},{:.1f},{:.1f},{:.3f},{:.3f},{:.2f}x\n",
                       size.name, size.batch, size.dim, n, cores_per_partition,
                       naive_ms, partitioned_ms, solo_ms,
                       naive_throughput, partitioned_throughput,
                       naive_congestion, partitioned_congestion, speedup);
        }
        fmt::print("#\n");
    }

    fmt::print("# Analysis:\n");
    fmt::print("# - partitioned_congestion > naive_congestion: Core partitioning helps\n");
    fmt::print("# - speedup > 1.0: Partitioned approach faster than naive\n");
    fmt::print("# - If partitioned ~= naive: Core grid doesn't affect scheduling\n");
    fmt::print("# - If both have low congestion: DRAM bandwidth is the bottleneck\n");

    device->close();
    return 0;
}
