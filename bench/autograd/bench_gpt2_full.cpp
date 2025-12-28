// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Full GPT-2 Training Benchmark (with Embeddings)
// Fair comparison with TTML nano_gpt - includes all components:
//   - Token embedding
//   - Positional embedding
//   - N transformer layers (LN + Attention + LN + FFN with GELU)
//   - Output projection
//
// Usage:
//   make bench-gpt2-full
//   BENCH_LAYERS=6 make bench-gpt2-full-run

#include "traced/transformer_train.hpp"
#include "common.hpp"

#include <ttnn/operations/trace.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>
#include <chrono>
#include <cstdlib>
#include <random>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;
using Tensor = tt::tt_metal::Tensor;

// DeviceGuard with configurable trace region
struct ConfigurableDeviceGuard {
    std::shared_ptr<MeshDevice> device;

    ConfigurableDeviceGuard(size_t trace_region_mb)
        : device(MeshDevice::create_unit_mesh(
            0,
            DEFAULT_L1_SMALL_SIZE,
            trace_region_mb * 1024 * 1024,
            1,
            DispatchCoreConfig{}
        )) {}

    ~ConfigurableDeviceGuard() {
        tt::tt_metal::distributed::Finish(device->mesh_command_queue());
        ttnn::close_device(*device);
    }

    MeshDevice& get() { return *device; }
};

constexpr int N_WARMUP = 3;
constexpr int N_ITERS = 10;

// TTML nano_gpt defaults
constexpr uint32_t DEFAULT_VOCAB = 256;
constexpr uint32_t DEFAULT_BATCH = 32;
constexpr uint32_t DEFAULT_SEQ = 256;
constexpr uint32_t DEFAULT_DIM = 384;
constexpr uint32_t DEFAULT_HEADS = 6;
constexpr uint32_t DEFAULT_FFN_MULT = 4;
constexpr uint32_t DEFAULT_LAYERS = 6;

// Create random token indices
Tensor create_random_tokens(uint32_t batch, uint32_t seq, uint32_t vocab, MeshDevice& dev) {
    std::vector<uint32_t> tokens(batch * seq);
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, vocab - 1);
    for (auto& t : tokens) {
        t = dist(rng);
    }
    return traced::make_indices(tokens, ttnn::Shape({batch, seq}), dev);
}

template<size_t N>
double benchmark_with_embedding(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t heads,
                                 uint32_t ffn_mult, uint32_t vocab, float lr, MeshDevice& dev) {
    // Create random token input
    auto token_ids = create_random_tokens(batch, seq, vocab, dev);

    // Create target logits (random for benchmarking)
    auto target = traced::make_full(ttnn::Shape({batch, seq, vocab}), 0.0f, dev);

    // Create full model with embeddings
    traced::TracedTransformerWithEmbedding<N> model(batch, seq, dim, heads, ffn_mult, vocab, lr, dev);

    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        model.train_step(token_ids, target);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; ++i) {
        model.train_step(token_ids, target);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
}

int main() {
    // Configuration from environment (TTML nano_gpt defaults)
    uint32_t batch = DEFAULT_BATCH;
    uint32_t seq = DEFAULT_SEQ;
    uint32_t dim = DEFAULT_DIM;
    uint32_t heads = DEFAULT_HEADS;
    uint32_t ffn_mult = DEFAULT_FFN_MULT;
    uint32_t vocab = DEFAULT_VOCAB;
    uint32_t num_layers = DEFAULT_LAYERS;
    float lr = 0.01f;
    size_t trace_mb = 128;

    if (const char* env = std::getenv("BENCH_BATCH")) batch = std::stoul(env);
    if (const char* env = std::getenv("BENCH_SEQ")) seq = std::stoul(env);
    if (const char* env = std::getenv("BENCH_DIM")) dim = std::stoul(env);
    if (const char* env = std::getenv("BENCH_HEADS")) heads = std::stoul(env);
    if (const char* env = std::getenv("BENCH_FFN_MULT")) ffn_mult = std::stoul(env);
    if (const char* env = std::getenv("BENCH_VOCAB")) vocab = std::stoul(env);
    if (const char* env = std::getenv("BENCH_LAYERS")) num_layers = std::stoul(env);
    if (const char* env = std::getenv("BENCH_TRACE_MB")) trace_mb = std::stoul(env);

    fmt::print("# Full GPT-2 Training Benchmark (with Embeddings)\n");
    fmt::print("# Config: batch={}, seq={}, dim={}, heads={}, ffn_mult={}, vocab={}, layers={}\n",
               batch, seq, dim, heads, ffn_mult, vocab, num_layers);
    fmt::print("# Trace region: {} MB, Warmup: {}, Iterations: {}\n\n",
               trace_mb, N_WARMUP, N_ITERS);

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

    // Open device
    fmt::print("Opening device with {} MB trace region...\n", trace_mb);
    ConfigurableDeviceGuard dg(trace_mb);
    auto& device = dg.get();
    fmt::print("Device ready\n\n");

    double our_ms = 0.0;

    // Benchmark with embeddings
    fmt::print("## Training step ({} layers, with embeddings)\n", num_layers);
    try {
        switch (num_layers) {
            case 1: our_ms = benchmark_with_embedding<1>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 2: our_ms = benchmark_with_embedding<2>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 3: our_ms = benchmark_with_embedding<3>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 6: our_ms = benchmark_with_embedding<6>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
        }
        fmt::print("Our implementation: {:.3f} ms/step, {:.3f} ms/layer\n\n",
                   our_ms, our_ms / num_layers);
    } catch (const std::exception& e) {
        fmt::print("FAILED: {}\n\n", e.what());
        return 1;
    }

    // TTML comparison
    const double ttml_ms = 510.0;  // TTML nano_gpt measured time
    double speedup = ttml_ms / our_ms;

    fmt::print("## Summary (Full GPT-2 Comparison)\n");
    fmt::print("| Implementation | Time (ms) | vs TTML |\n");
    fmt::print("|----------------|-----------|--------|\n");
    fmt::print("| TTML nano_gpt  | {:9.3f} | {:6.2f}x |\n", ttml_ms, 1.0);
    fmt::print("| Ours (w/ emb)  | {:9.3f} | {:6.2f}x |\n", our_ms, speedup);

    fmt::print("\n## Architecture Match\n");
    fmt::print("Both include:\n");
    fmt::print("  - Token embedding [vocab={}, dim={}]\n", vocab, dim);
    fmt::print("  - Positional embedding [seq={}, dim={}]\n", seq, dim);
    fmt::print("  - {} x Transformer layers (LN + Attention + LN + GELU FFN)\n", num_layers);
    fmt::print("  - Output projection [dim={}, vocab={}]\n", dim, vocab);

    fmt::print("\n## Configuration\n");
    fmt::print("batch={}, seq={}, dim={}, heads={}, ffn_mult={}, vocab={}, layers={}\n",
               batch, seq, dim, heads, ffn_mult, vocab, num_layers);

    // CSV output
    fmt::print("\n# CSV: batch,seq,dim,heads,vocab,layers,our_ms,ttml_ms,speedup\n");
    fmt::print("# {},{},{},{},{},{},{:.3f},{:.3f},{:.2f}\n",
               batch, seq, dim, heads, vocab, num_layers,
               our_ms, ttml_ms, speedup);

    return 0;
}
