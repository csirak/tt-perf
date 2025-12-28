// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: Memory Config & Fusion for Matmul+ReLU
// Compares four approaches:
// 1. DRAM (default) - intermediate tensor goes to DRAM
// 2. L1 Interleaved - intermediate in L1 but spread across cores
// 3. L1 Height Sharded - explicit per-core sharding in L1
// 4. Fused - kernel-level fusion via matmul activation parameter

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/memory_config/memory_config.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <tt-metalium/distributed.hpp>

#include <chrono>
#include <vector>
#include <numeric>

using namespace ttnn;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

constexpr int N_WARMUP = 10;
constexpr int N_TIMED = 50;

struct BenchConfig {
    const char* name;
    uint32_t batch;
    uint32_t dim;
};

// Memory configs
const MemoryConfig L1_INTERLEAVED{TensorMemoryLayout::INTERLEAVED, BufferType::L1};
const MemoryConfig DRAM_INTERLEAVED{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};

// 1. DRAM (baseline) - default memory config
double bench_dram(MeshDevice& dev, const BenchConfig& cfg) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    auto x = ones(Shape({cfg.batch, cfg.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto w = ones(Shape({cfg.dim, cfg.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    Finish(cq);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        auto h = ttnn::matmul(x, w);  // Output to DRAM (default)
        auto out = ttnn::relu(h);      // Output to DRAM (default)
    }
    Finish(cq);

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        auto h = ttnn::matmul(x, w);
        auto out = ttnn::relu(h);
        Finish(cq);

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

// 2. L1 Interleaved - explicit L1 memory config (spread across cores)
double bench_l1_interleaved(MeshDevice& dev, const BenchConfig& cfg) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    auto x = ones(Shape({cfg.batch, cfg.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto w = ones(Shape({cfg.dim, cfg.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    Finish(cq);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        auto h = ttnn::matmul(x, w, false, false, L1_INTERLEAVED);
        auto out = ttnn::relu(h, L1_INTERLEAVED);
    }
    Finish(cq);

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        auto h = ttnn::matmul(x, w, false, false, L1_INTERLEAVED);
        auto out = ttnn::relu(h, L1_INTERLEAVED);
        Finish(cq);

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

// 3. Fused via matmul activation parameter
// Uses ttnn::matmul with activation="relu" which triggers PACK_RELU in the kernel
double bench_fused_matmul(MeshDevice& dev, const BenchConfig& cfg) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    auto x = ones(Shape({cfg.batch, cfg.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto w = ones(Shape({cfg.dim, cfg.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    Finish(cq);

    // Use matmul with activation parameter for true fusion
    // matmul signature: (a, b, transpose_a, transpose_b, memory_config, dtype, program_config, activation, ...)
    std::string activation = "relu";

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        auto out = ttnn::matmul(x, w, false, false, L1_INTERLEAVED, std::nullopt, std::nullopt, activation);
    }
    Finish(cq);

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        auto out = ttnn::matmul(x, w, false, false, L1_INTERLEAVED, std::nullopt, std::nullopt, activation);
        Finish(cq);

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

// 4. Fused with DRAM output (to isolate fusion benefit from L1 benefit)
double bench_fused_dram(MeshDevice& dev, const BenchConfig& cfg) {
    MeshCommandQueue& cq = dev.mesh_command_queue();

    auto x = ones(Shape({cfg.batch, cfg.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    auto w = ones(Shape({cfg.dim, cfg.dim}), DataType::BFLOAT16, TILE_LAYOUT, dev);
    Finish(cq);

    std::string activation = "relu";

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        auto out = ttnn::matmul(x, w, false, false, DRAM_INTERLEAVED, std::nullopt, std::nullopt, activation);
    }
    Finish(cq);

    // Timed runs
    std::vector<double> times_ms(N_TIMED);
    for (int t = 0; t < N_TIMED; t++) {
        auto start = std::chrono::high_resolution_clock::now();

        auto out = ttnn::matmul(x, w, false, false, DRAM_INTERLEAVED, std::nullopt, std::nullopt, activation);
        Finish(cq);

        auto end = std::chrono::high_resolution_clock::now();
        times_ms[t] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / N_TIMED;
}

int main() {
    auto device = MeshDevice::create_unit_mesh(0);

    // Size configurations (all dimensions tile-aligned)
    // Include smaller sizes to see where L1 helps more
    std::vector<BenchConfig> configs = {
        {"tiny",   32,   128},   // 8KB output
        {"small",  32,   256},   // 32KB output
        {"medium", 256,  512},   // 512KB output
        {"large",  512,  1024},  // 2MB output
        {"xlarge", 1024, 1024},  // 4MB output (x has 1024 batch)
    };

    fmt::print("# Matmul+ReLU Memory Config & Fusion Benchmark\n");
    fmt::print("# Network: x[B,D] @ W[D,D] -> ReLU\n");
    fmt::print("# Warmup: {}, Timed iterations: {}\n", N_WARMUP, N_TIMED);
    fmt::print("#\n");
    fmt::print("# DRAM:       matmul->DRAM, relu->DRAM (2 ops, 2 DRAM writes)\n");
    fmt::print("# L1_INT:     matmul->L1, relu->L1 (2 ops, L1 interleaved)\n");
    fmt::print("# FUSED_DRAM: matmul+relu fused->DRAM (1 op, PACK_RELU)\n");
    fmt::print("# FUSED_L1:   matmul+relu fused->L1 (1 op, PACK_RELU, L1 output)\n");
    fmt::print("#\n");
    fmt::print("size,batch,dim,dram_ms,l1_int_ms,fused_dram_ms,fused_l1_ms,fusion_speedup,best_speedup\n");

    for (const auto& cfg : configs) {
        double dram_ms = bench_dram(*device, cfg);
        double l1_int_ms = bench_l1_interleaved(*device, cfg);
        double fused_dram_ms = bench_fused_dram(*device, cfg);
        double fused_l1_ms = bench_fused_matmul(*device, cfg);

        // Fusion speedup: compare fused_dram vs dram (isolates fusion benefit)
        double fusion_speedup = dram_ms / fused_dram_ms;
        // Best speedup: compare fused_l1 vs dram (fusion + L1 combined)
        double best_speedup = dram_ms / fused_l1_ms;

        fmt::print("{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.2f}x,{:.2f}x\n",
                   cfg.name, cfg.batch, cfg.dim,
                   dram_ms, l1_int_ms, fused_dram_ms, fused_l1_ms,
                   fusion_speedup, best_speedup);
    }

    device->close();
    return 0;
}
