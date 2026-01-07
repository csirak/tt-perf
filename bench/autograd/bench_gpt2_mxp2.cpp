// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GPT-2 MXP2 Benchmark: Batched Weight Typecast
//
// Compares:
// 1. MXP: Batched activation typecast, INLINE weight typecast in sgd_step
// 2. MXP2: Batched activation typecast, BATCHED weight typecast after sgd_step
//
// Usage:
//   make bench-gpt2-mxp2
//   BENCH_LAYERS=6 make bench-gpt2-mxp2-run

#include "autograd/nn.hpp"

#include <ttnn/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>
#include <chrono>
#include <cstdlib>
#include <string>

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

constexpr int DEFAULT_WARMUP = 3;
constexpr int DEFAULT_ITERS = 10;

// Tile-aligned config
constexpr uint32_t DEFAULT_BATCH = 32;
constexpr uint32_t DEFAULT_SEQ = 256;
constexpr uint32_t DEFAULT_DIM = 512;
constexpr uint32_t DEFAULT_HEADS = 8;
constexpr uint32_t DEFAULT_FFN_MULT = 4;
constexpr uint32_t DEFAULT_VOCAB = 256;
constexpr uint32_t DEFAULT_LAYERS = 6;

// MXP benchmark (batched activation typecast, inline weight typecast)
template<size_t N>
double benchmark_mxp(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t heads,
                     uint32_t ffn_mult, uint32_t vocab, float lr, int warmup, int iters,
                     MeshDevice& dev) {
    PersistentGPT2MXP<N> model(batch, seq, dim, heads, ffn_mult, vocab, lr, dev);
    bool cache_graph = std::getenv("MXP_DISABLE_GRAPH_CACHE") == nullptr;
    if (cache_graph) {
        model.build_graph();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    for (int i = 0; i < warmup; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

// MXP2 benchmark (batched activation typecast, batched weight typecast)
template<size_t N>
double benchmark_mxp2(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t heads,
                      uint32_t ffn_mult, uint32_t vocab, float lr, int warmup, int iters,
                      MeshDevice& dev) {
    PersistentGPT2MXP2<N> model(batch, seq, dim, heads, ffn_mult, vocab, lr, dev);
    bool cache_graph = std::getenv("MXP_DISABLE_GRAPH_CACHE") == nullptr;
    if (cache_graph) {
        model.build_graph();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    for (int i = 0; i < warmup; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
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
    int warmup = DEFAULT_WARMUP;
    int iters = DEFAULT_ITERS;
    bool run_mxp = true;
    bool run_mxp2 = true;

    if (const char* env = std::getenv("BENCH_BATCH")) batch = std::stoul(env);
    if (const char* env = std::getenv("BENCH_SEQ")) seq = std::stoul(env);
    if (const char* env = std::getenv("BENCH_DIM")) dim = std::stoul(env);
    if (const char* env = std::getenv("BENCH_HEADS")) heads = std::stoul(env);
    if (const char* env = std::getenv("BENCH_FFN_MULT")) ffn_mult = std::stoul(env);
    if (const char* env = std::getenv("BENCH_VOCAB")) vocab = std::stoul(env);
    if (const char* env = std::getenv("BENCH_LAYERS")) num_layers = std::stoul(env);
    if (const char* env = std::getenv("BENCH_WARMUP")) warmup = std::stoi(env);
    if (const char* env = std::getenv("BENCH_ITERS")) iters = std::stoi(env);
    if (const char* env = std::getenv("BENCH_ONLY")) {
        std::string only = env;
        if (only == "mxp") {
            run_mxp2 = false;
        } else if (only == "mxp2") {
            run_mxp = false;
        }
    }

    fmt::print("# GPT-2 MXP2 Benchmark: Batched Weight Typecast\n");
    fmt::print("# Comparing: MXP (inline weight typecast) vs MXP2 (batched weight typecast)\n");
    fmt::print("# Config: batch={}, seq={}, dim={}, heads={}, ffn_mult={}, vocab={}, layers={}\n",
               batch, seq, dim, heads, ffn_mult, vocab, num_layers);
    fmt::print("# Warmup: {}, Iterations: {}\n\n", warmup, iters);

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

    double mxp_ms = -1.0, mxp2_ms = -1.0;

    // Test 1: MXP (inline weight typecast)
    if (run_mxp) {
        fmt::print("## Test 1: MXP - Inline weight typecast ({} layers)\n", num_layers);
        try {
            switch (num_layers) {
                case 1: mxp_ms = benchmark_mxp<1>(batch, seq, dim, heads, ffn_mult, vocab, lr, warmup, iters, device); break;
                case 2: mxp_ms = benchmark_mxp<2>(batch, seq, dim, heads, ffn_mult, vocab, lr, warmup, iters, device); break;
                case 3: mxp_ms = benchmark_mxp<3>(batch, seq, dim, heads, ffn_mult, vocab, lr, warmup, iters, device); break;
                case 6: mxp_ms = benchmark_mxp<6>(batch, seq, dim, heads, ffn_mult, vocab, lr, warmup, iters, device); break;
            }
            fmt::print("MXP: {:.3f} ms/step, {:.3f} ms/layer\n\n", mxp_ms, mxp_ms / num_layers);
        } catch (const std::exception& e) {
            fmt::print("MXP FAILED: {}\n\n", e.what());
            return 1;
        }
    }

    // Test 2: MXP2 (batched weight typecast)
    if (run_mxp2) {
        fmt::print("## Test 2: MXP2 - Batched weight typecast ({} layers)\n", num_layers);
        try {
            switch (num_layers) {
                case 1: mxp2_ms = benchmark_mxp2<1>(batch, seq, dim, heads, ffn_mult, vocab, lr, warmup, iters, device); break;
                case 2: mxp2_ms = benchmark_mxp2<2>(batch, seq, dim, heads, ffn_mult, vocab, lr, warmup, iters, device); break;
                case 3: mxp2_ms = benchmark_mxp2<3>(batch, seq, dim, heads, ffn_mult, vocab, lr, warmup, iters, device); break;
                case 6: mxp2_ms = benchmark_mxp2<6>(batch, seq, dim, heads, ffn_mult, vocab, lr, warmup, iters, device); break;
            }
            fmt::print("MXP2: {:.3f} ms/step, {:.3f} ms/layer\n\n", mxp2_ms, mxp2_ms / num_layers);
        } catch (const std::exception& e) {
            fmt::print("MXP2 FAILED: {}\n\n", e.what());
            return 1;
        }
    }

    // Calculate FLOPS
    uint32_t ffn_dim = dim * ffn_mult;
    uint64_t head_dim = dim / heads;
    uint64_t B = batch, S = seq, D = dim, H = heads, d = head_dim, F = ffn_dim;
    uint64_t layer_flops_fwd = 8*B*S*D*D + 4*B*H*S*S*d + 4*B*S*D*F;
    uint64_t layer_flops_total = 3 * layer_flops_fwd;
    uint64_t model_flops = num_layers * layer_flops_total + 3 * 2 * B * S * D * vocab;

    double mxp_tflops = mxp_ms > 0.0 ? (model_flops / 1e12) / (mxp_ms / 1000.0) : 0.0;
    double mxp2_tflops = mxp2_ms > 0.0 ? (model_flops / 1e12) / (mxp2_ms / 1000.0) : 0.0;
    double mxp2_vs_mxp = (mxp_ms > 0.0 && mxp2_ms > 0.0) ? (mxp_ms / mxp2_ms) : 0.0;

    // Summary
    fmt::print("## Summary\n");
    fmt::print("| Mode | Time (ms) | TFLOPS | vs MXP |\n");
    fmt::print("|------|-----------|--------|--------|\n");
    if (mxp_ms > 0.0) {
        fmt::print("| MXP (inline weight typecast) | {:.3f} | {:.2f} | 1.00x |\n", mxp_ms, mxp_tflops);
    } else {
        fmt::print("| MXP (inline weight typecast) | N/A | N/A | N/A |\n");
    }
    if (mxp2_ms > 0.0 && mxp_ms > 0.0) {
        fmt::print("| MXP2 (batched weight typecast) | {:.3f} | {:.2f} | {:.2f}x |\n", mxp2_ms, mxp2_tflops, mxp2_vs_mxp);
    } else if (mxp2_ms > 0.0) {
        fmt::print("| MXP2 (batched weight typecast) | {:.3f} | {:.2f} | N/A |\n", mxp2_ms, mxp2_tflops);
    } else {
        fmt::print("| MXP2 (batched weight typecast) | N/A | N/A | N/A |\n");
    }
    fmt::print("\n");

    // Analysis
    fmt::print("## Analysis\n");
    if (mxp2_ms > 0.0 && mxp_ms > 0.0 && mxp2_vs_mxp > 1.02) {
        fmt::print("MXP2 provides {:.1f}% speedup over MXP (batched weight typecast helps!)\n", (mxp2_vs_mxp - 1.0) * 100);
    } else if (mxp2_ms > 0.0 && mxp_ms > 0.0 && mxp2_vs_mxp < 0.98) {
        fmt::print("MXP2 is {:.1f}% SLOWER than MXP (batching overhead)\n", (1.0 - mxp2_vs_mxp) * 100);
    } else if (mxp2_ms > 0.0 && mxp_ms > 0.0) {
        fmt::print("MXP2 ~same as MXP (weight typecast already pipelined)\n");
    } else {
        fmt::print("MXP2 comparison skipped\n");
    }

    fmt::print("\n# CSV: batch,seq,dim,heads,vocab,layers,mxp_ms,mxp2_ms\n");
    fmt::print("# {},{},{},{},{},{},{:.3f},{:.3f}\n",
               batch, seq, dim, heads, vocab, num_layers,
               mxp_ms > 0.0 ? mxp_ms : 0.0,
               mxp2_ms > 0.0 ? mxp2_ms : 0.0);

    return 0;
}
