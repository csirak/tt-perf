// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GPT-2 Training: Static vs Traced Benchmark
// Compares static (non-traced) vs traced execution for full GPT-2 with embeddings.
// Internal comparison only - measures tracing benefit for our implementation.
//
// Usage:
//   make bench-gpt2-trace
//   BENCH_LAYERS=6 make bench-gpt2-trace-run

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
double benchmark_static(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t heads,
                        uint32_t ffn_mult, uint32_t vocab, float lr, MeshDevice& dev) {
    // Create random token input
    auto token_ids = create_random_tokens(batch, seq, vocab, dev);

    // Create target logits
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

template<size_t N>
double benchmark_traced(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t heads,
                        uint32_t ffn_mult, uint32_t vocab, float lr, MeshDevice& dev) {
    // Create random token input
    auto token_ids = create_random_tokens(batch, seq, vocab, dev);

    // Create target logits
    auto target = traced::make_full(ttnn::Shape({batch, seq, vocab}), 0.0f, dev);

    // Create full model with embeddings
    traced::TracedTransformerWithEmbedding<N> model(batch, seq, dim, heads, ffn_mult, vocab, lr, dev);

    // Non-traced warmup first (compile kernels)
    for (int i = 0; i < 2; ++i) {
        model.train_step(token_ids, target);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Capture trace
    auto trace_id = ttnn::operations::trace::begin_trace_capture(&dev, std::nullopt);
    model.train_step(token_ids, target);
    ttnn::operations::trace::end_trace_capture(&dev, trace_id, std::nullopt);

    // Traced warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        ttnn::operations::trace::execute_trace(&dev, trace_id, std::nullopt, false);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; ++i) {
        ttnn::operations::trace::execute_trace(&dev, trace_id, std::nullopt, false);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    ttnn::operations::trace::release_trace(&dev, trace_id);

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

    fmt::print("# GPT-2 Training: Static vs Traced\n");
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

    double static_ms = 0.0, traced_ms = 0.0;
    bool trace_success = false;

    // =========================================================================
    // Test 1: Static training step
    // =========================================================================
    fmt::print("## Test 1: Static training step ({} layers)\n", num_layers);
    try {
        switch (num_layers) {
            case 1: static_ms = benchmark_static<1>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 2: static_ms = benchmark_static<2>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 3: static_ms = benchmark_static<3>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 6: static_ms = benchmark_static<6>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
        }
        fmt::print("Static: {:.3f} ms/step, {:.3f} ms/layer\n\n",
                   static_ms, static_ms / num_layers);
    } catch (const std::exception& e) {
        fmt::print("Static FAILED: {}\n\n", e.what());
        return 1;
    }

    // =========================================================================
    // Test 2: Traced training step
    // =========================================================================
    fmt::print("## Test 2: Traced training step ({} layers)\n", num_layers);
    try {
        switch (num_layers) {
            case 1: traced_ms = benchmark_traced<1>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 2: traced_ms = benchmark_traced<2>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 3: traced_ms = benchmark_traced<3>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 6: traced_ms = benchmark_traced<6>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
        }
        trace_success = true;
        fmt::print("Traced: {:.3f} ms/step, {:.3f} ms/layer\n\n",
                   traced_ms, traced_ms / num_layers);
    } catch (const std::exception& e) {
        fmt::print("Traced FAILED: {}\n\n", e.what());
    }

    // =========================================================================
    // Summary
    // =========================================================================
    fmt::print("## Summary\n");
    fmt::print("| Mode     | Total (ms) | Per-layer (ms) | Speedup |\n");
    fmt::print("|----------|------------|----------------|--------|\n");
    fmt::print("| Static   | {:10.3f} | {:14.3f} | {:6.2f}x |\n",
               static_ms, static_ms / num_layers, 1.0);
    if (trace_success) {
        double speedup = static_ms / traced_ms;
        fmt::print("| Traced   | {:10.3f} | {:14.3f} | {:6.2f}x |\n",
                   traced_ms, traced_ms / num_layers, speedup);
    } else {
        fmt::print("| Traced   |       FAIL |           FAIL |   FAIL |\n");
    }

    fmt::print("\n## Configuration\n");
    fmt::print("batch={}, seq={}, dim={}, heads={}, ffn_mult={}, vocab={}, layers={}\n",
               batch, seq, dim, heads, ffn_mult, vocab, num_layers);

    // CSV output
    fmt::print("\n# CSV: batch,seq,dim,heads,vocab,layers,static_ms,traced_ms,speedup\n");
    if (trace_success) {
        fmt::print("# {},{},{},{},{},{},{:.3f},{:.3f},{:.2f}\n",
                   batch, seq, dim, heads, vocab, num_layers,
                   static_ms, traced_ms, static_ms / traced_ms);
    } else {
        fmt::print("# {},{},{},{},{},{},{:.3f},FAIL,FAIL\n",
                   batch, seq, dim, heads, vocab, num_layers, static_ms);
    }

    return trace_success ? 0 : 1;
}
