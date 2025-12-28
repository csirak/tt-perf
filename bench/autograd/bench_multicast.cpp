// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: Parallel MLPs with shared weights (multicast-friendly)
// Compares:
// 1. Naive: Each MLP instance creates its own weight tensors (N×2 DRAM loads)
// 2. Shared: All MLP instances share the same weight tensors (1×2 DRAM loads)
//
// The shared approach should benefit from:
// - Reduced DRAM bandwidth (weights loaded once, cached/multicast)
// - Better L1 utilization (no duplicate weight copies)

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

// L1 interleaved memory config for better caching
const MemoryConfig L1_INTERLEAVED{TensorMemoryLayout::INTERLEAVED, BufferType::L1};

// Activation tensors for one MLP instance
struct MLPActivations {
    Tensor x;
    Tensor dout;
};

// Shared weights for all MLP instances
struct MLPWeights {
    Tensor w1, b1, w2, b2;
};

// Create shared weights (one copy for all MLPs)
MLPWeights create_shared_weights(MeshDevice& dev, uint32_t dim) {
    return {
        ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev, L1_INTERLEAVED),
        zeros(Shape({1, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev, L1_INTERLEAVED),
        ones(Shape({dim, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev, L1_INTERLEAVED),
        zeros(Shape({1, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev, L1_INTERLEAVED),
    };
}

// Create activations for one MLP instance
MLPActivations create_activations(MeshDevice& dev, uint32_t batch, uint32_t dim) {
    return {
        ones(Shape({batch, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        ones(Shape({batch, dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
    };
}

// Run forward+backward for one MLP with given weights (non-blocking)
void run_mlp_shared(const MLPActivations& act, const MLPWeights& wt) {
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

// Benchmark: Naive approach - each MLP has its own weights
double bench_naive(MeshDevice& dev, const SizeConfig& size, int n_parallel) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    // Each MLP instance has its own weights AND activations
    struct FullMLP {
        Tensor x, w1, b1, w2, b2, dout;
    };

    std::vector<FullMLP> mlps;
    for (int i = 0; i < n_parallel; i++) {
        mlps.push_back({
            ones(Shape({size.batch, size.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
            ones(Shape({size.dim, size.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
            zeros(Shape({1, size.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
            ones(Shape({size.dim, size.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
            zeros(Shape({1, size.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
            ones(Shape({size.batch, size.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev),
        });
    }
    Finish(cq);

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        for (const auto& m : mlps) {
            // Forward
            auto mm1 = ttnn::matmul(m.x, m.w1, false, true);
            auto h_pre = ttnn::add(mm1, m.b1);
            auto h = ttnn::relu(h_pre);
            auto mm2 = ttnn::matmul(h, m.w2, false, true);
            auto out = ttnn::add(mm2, m.b2);
            // Backward
            auto dh = ttnn::matmul(m.dout, m.w2, false, false);
            auto relu_mask = ttnn::gtz(h_pre);
            auto dh_pre = ttnn::multiply(dh, relu_mask);
            auto dx = ttnn::matmul(dh_pre, m.w1, false, false);
        }
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        for (const auto& m : mlps) {
            auto mm1 = ttnn::matmul(m.x, m.w1, false, true);
            auto h_pre = ttnn::add(mm1, m.b1);
            auto h = ttnn::relu(h_pre);
            auto mm2 = ttnn::matmul(h, m.w2, false, true);
            auto out = ttnn::add(mm2, m.b2);
            auto dh = ttnn::matmul(m.dout, m.w2, false, false);
            auto relu_mask = ttnn::gtz(h_pre);
            auto dh_pre = ttnn::multiply(dh, relu_mask);
            auto dx = ttnn::matmul(dh_pre, m.w1, false, false);
        }
        Finish(cq);

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

// Benchmark: Shared weights - all MLPs use same weight tensors
double bench_shared(MeshDevice& dev, const SizeConfig& size, int n_parallel) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    // Create shared weights (one copy in L1)
    MLPWeights weights = create_shared_weights(dev, size.dim);

    // Create activations for each MLP instance
    std::vector<MLPActivations> activations;
    for (int i = 0; i < n_parallel; i++) {
        activations.push_back(create_activations(dev, size.batch, size.dim));
    }
    Finish(cq);

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        for (const auto& act : activations) {
            run_mlp_shared(act, weights);
        }
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        for (const auto& act : activations) {
            run_mlp_shared(act, weights);
        }
        Finish(cq);

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

// Benchmark: Solo MLP (baseline for congestion ratio)
double bench_solo(MeshDevice& dev, const SizeConfig& size) {
    return bench_shared(dev, size, 1);
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

    std::vector<int> parallel_counts = {1, 2, 4, 8};

    fmt::print("# Shared Weights vs Naive Parallel MLP Benchmark\n");
    fmt::print("# Naive: Each MLP creates own weights (N×2 DRAM loads per forward)\n");
    fmt::print("# Shared: All MLPs share same weights in L1 (1×2 loads, potential multicast)\n");
    fmt::print("# Warmup: {}, Timed iterations: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");
    fmt::print("size,batch,dim,n_parallel,naive_ms,shared_ms,speedup,naive_congestion,shared_congestion\n");

    for (const auto& size : sizes) {
        // Get solo baseline
        double solo_ms = bench_solo(*device, size);
        double solo_throughput = size.batch / solo_ms * 1000.0;

        for (int n : parallel_counts) {
            double naive_ms = bench_naive(*device, size, n);
            double shared_ms = bench_shared(*device, size, n);

            double speedup = naive_ms / shared_ms;

            // Congestion ratios
            double naive_throughput = (n * size.batch) / naive_ms * 1000.0;
            double shared_throughput = (n * size.batch) / shared_ms * 1000.0;
            double naive_congestion = naive_throughput / (n * solo_throughput);
            double shared_congestion = shared_throughput / (n * solo_throughput);

            fmt::print("{},{},{},{},{:.3f},{:.3f},{:.2f}x,{:.3f},{:.3f}\n",
                       size.name, size.batch, size.dim, n,
                       naive_ms, shared_ms, speedup,
                       naive_congestion, shared_congestion);
        }
        fmt::print("#\n");
    }

    fmt::print("# Analysis:\n");
    fmt::print("# - speedup > 1.0: Shared weights faster than naive\n");
    fmt::print("# - shared_congestion > naive_congestion: Better scaling with shared weights\n");
    fmt::print("# - Key benefit: Reduced DRAM bandwidth from weight sharing\n");

    device->close();
    return 0;
}
