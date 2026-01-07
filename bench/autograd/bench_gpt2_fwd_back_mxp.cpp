// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GPT-2 Mixed Precision Benchmark: BF16 vs BFP8 vs MXP (batched typecast)
// Compares three implementations:
// 1. BF16: Pure BF16 baseline
// 2. BFP8: Current mixed precision (inline typecast in backward)
// 3. MXP: Batched typecast before backward (optimized)
//
// Usage:
//   make bench-gpt2-fwd-back-mxp
//   BENCH_LAYERS=6 make bench-gpt2-fwd-back-mxp-run

#include "static-mxp/nn.hpp"

#include <ttnn/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>
#include <chrono>
#include <cstdlib>

using namespace static_autograd;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;
using DispatchCoreType = tt::tt_metal::DispatchCoreType;

// DeviceGuard with ETH dispatch (default for N300 8x8 grid)
struct DeviceGuard {
    std::shared_ptr<MeshDevice> device;

    DeviceGuard()
        : device([]() {
            bool use_worker = std::getenv("USE_WORKER_DISPATCH") != nullptr;
            return MeshDevice::create_unit_mesh(
                0,
                DEFAULT_L1_SMALL_SIZE,
                0,  // No trace region needed
                1,
                use_worker ? DispatchCoreConfig{}
                           : DispatchCoreConfig{DispatchCoreType::ETH}
            );
        }()) {
        auto grid = device->compute_with_storage_grid_size();
        fmt::print("# Compute grid: {}x{} = {} cores\n", grid.x, grid.y, grid.x * grid.y);
    }

    ~DeviceGuard() {
        tt::tt_metal::distributed::Finish(device->mesh_command_queue());
        ttnn::close_device(*device);
    }

    MeshDevice& get() { return *device; }
};

constexpr int N_WARMUP = 3;
constexpr int N_ITERS = 10;

// Tile-aligned config
constexpr uint32_t DEFAULT_BATCH = 32;
constexpr uint32_t DEFAULT_SEQ = 256;
constexpr uint32_t DEFAULT_DIM = 512;
constexpr uint32_t DEFAULT_HEADS = 8;
constexpr uint32_t DEFAULT_FFN_MULT = 4;
constexpr uint32_t DEFAULT_VOCAB = 256;
constexpr uint32_t DEFAULT_LAYERS = 6;

// BF16 benchmark (baseline)
template<size_t N>
double benchmark_bf16(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t heads,
                      uint32_t ffn_mult, uint32_t vocab, float lr, MeshDevice& dev) {
    PersistentGPT2<N> model(batch, seq, dim, heads, ffn_mult, vocab, lr, dev);
    model.build();
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    for (int i = 0; i < N_WARMUP; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
}

// Mixed precision benchmark: BF16 attention + BFP8 FFN
template<size_t N>
double benchmark_mixed(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t heads,
                       uint32_t ffn_mult, uint32_t vocab, float lr, MeshDevice& dev) {
    PersistentGPT2BFP8<N> model(batch, seq, dim, heads, ffn_mult, vocab, lr, dev);
    model.build();
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    for (int i = 0; i < N_WARMUP; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
}

int main() {
    // Configuration from environment
    uint32_t batch = DEFAULT_BATCH;
    uint32_t seq = DEFAULT_SEQ;
    uint32_t dim = DEFAULT_DIM;
    uint32_t heads = DEFAULT_HEADS;
    uint32_t ffn_mult = DEFAULT_FFN_MULT;
    uint32_t vocab = DEFAULT_VOCAB;
    uint32_t num_layers = DEFAULT_LAYERS;
    float lr = 0.01f;

    if (const char* env = std::getenv("BENCH_BATCH")) batch = std::stoul(env);
    if (const char* env = std::getenv("BENCH_SEQ")) seq = std::stoul(env);
    if (const char* env = std::getenv("BENCH_DIM")) dim = std::stoul(env);
    if (const char* env = std::getenv("BENCH_HEADS")) heads = std::stoul(env);
    if (const char* env = std::getenv("BENCH_FFN_MULT")) ffn_mult = std::stoul(env);
    if (const char* env = std::getenv("BENCH_VOCAB")) vocab = std::stoul(env);
    if (const char* env = std::getenv("BENCH_LAYERS")) num_layers = std::stoul(env);

    fmt::print("# GPT-2 Mixed Precision Benchmark\n");
    fmt::print("# Config: batch={}, seq={}, dim={}, heads={}, ffn_mult={}, vocab={}, layers={}\n",
               batch, seq, dim, heads, ffn_mult, vocab, num_layers);
    fmt::print("# Warmup: {}, Iterations: {}\n\n", N_WARMUP, N_ITERS);

    // Validate dimensions
    if (dim % heads != 0) {
        fmt::print("ERROR: dim ({}) must be divisible by heads ({})\n", dim, heads);
        return 1;
    }
    if (batch % 32 != 0 || seq % 32 != 0 || dim % 32 != 0 || vocab % 32 != 0) {
        fmt::print("ERROR: batch, seq, dim, vocab must be multiples of 32\n");
        return 1;
    }
    if (num_layers != 1 && num_layers != 2 && num_layers != 3 && num_layers != 6) {
        fmt::print("ERROR: Supported layer counts: 1, 2, 3, 6\n");
        return 1;
    }

    fmt::print("Opening device...\n");
    DeviceGuard guard;
    MeshDevice& device = guard.get();
    fmt::print("Device ready\n\n");

    double bf16_ms = 0.0, mixed_ms = 0.0;

    // Test 1: Pure BF16
    fmt::print("## Test 1: Pure BF16 ({} layers)\n", num_layers);
    try {
        switch (num_layers) {
            case 1: bf16_ms = benchmark_bf16<1>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 2: bf16_ms = benchmark_bf16<2>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 3: bf16_ms = benchmark_bf16<3>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 6: bf16_ms = benchmark_bf16<6>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
        }
        fmt::print("BF16: {:.3f} ms/step, {:.3f} ms/layer\n\n", bf16_ms, bf16_ms / num_layers);
    } catch (const std::exception& e) {
        fmt::print("BF16 FAILED: {}\n\n", e.what());
        return 1;
    }

    // Test 2: Mixed Precision (BF16 Attention + BFP8 FFN)
    fmt::print("## Test 2: Mixed Precision - BF16 Attn + BFP8 FFN ({} layers)\n", num_layers);
    try {
        switch (num_layers) {
            case 1: mixed_ms = benchmark_mixed<1>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 2: mixed_ms = benchmark_mixed<2>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 3: mixed_ms = benchmark_mixed<3>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 6: mixed_ms = benchmark_mixed<6>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
        }
        fmt::print("Mixed: {:.3f} ms/step, {:.3f} ms/layer\n\n", mixed_ms, mixed_ms / num_layers);
    } catch (const std::exception& e) {
        fmt::print("Mixed FAILED: {}\n\n", e.what());
        return 1;
    }

    // Calculate FLOPS (same for both - compute is identical, just precision differs)
    uint32_t ffn_dim = dim * ffn_mult;
    uint64_t head_dim = dim / heads;
    uint64_t B = batch, S = seq, D = dim, H = heads, d = head_dim, F = ffn_dim;
    uint64_t layer_flops_fwd = 8*B*S*D*D + 4*B*H*S*S*d + 4*B*S*D*F;
    uint64_t layer_flops_total = 3 * layer_flops_fwd;
    uint64_t model_flops = num_layers * layer_flops_total + 3 * 2 * B * S * D * vocab;

    double bf16_tflops = (model_flops / 1e12) / (bf16_ms / 1000.0);
    double mixed_tflops = (model_flops / 1e12) / (mixed_ms / 1000.0);
    double speedup = bf16_ms / mixed_ms;

    // Summary
    fmt::print("## Summary\n");
    fmt::print("| Mode | Time (ms) | TFLOPS | vs BF16 |\n");
    fmt::print("|------|-----------|--------|--------|\n");
    fmt::print("| BF16 | {:.3f} | {:.2f} | 1.00x |\n", bf16_ms, bf16_tflops);
    fmt::print("| Mixed (BF16 Attn + BFP8 FFN) | {:.3f} | {:.2f} | {:.2f}x |\n",
               mixed_ms, mixed_tflops, speedup);
    fmt::print("\n");

    // Analysis
    fmt::print("## Analysis\n");
    if (speedup > 1.05) {
        fmt::print("✓ Mixed precision provides {:.1f}% speedup\n", (speedup - 1.0) * 100);
    } else if (speedup < 0.95) {
        fmt::print("✗ Mixed precision is {:.1f}% SLOWER - backward pass BF16 overhead\n", (1.0 - speedup) * 100);
    } else {
        fmt::print("△ Mixed precision ~same as BF16 (backward uses BF16)\n");
    }

    fmt::print("\n# CSV: batch,seq,dim,heads,vocab,layers,bf16_ms,mixed_ms,speedup\n");
    fmt::print("# {},{},{},{},{},{},{:.3f},{:.3f},{:.2f}\n",
               batch, seq, dim, heads, vocab, num_layers, bf16_ms, mixed_ms, speedup);

    return 0;
}
