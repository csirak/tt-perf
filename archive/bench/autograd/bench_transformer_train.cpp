// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Transformer Training Benchmark
// Measures full training step (forward + backward + SGD) for comparison with TTML.
//
// Usage:
//   make bench-transformer-train
//   BENCH_BATCH=32 BENCH_SEQ=256 BENCH_DIM=384 BENCH_HEADS=6 BENCH_LAYERS=6 make bench-transformer-train-run

#include "traced/transformer_train.hpp"
#include "traced/trace.hpp"
#include "common.hpp"

#include <ttnn/operations/trace.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>
#include <chrono>
#include <cstdlib>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;

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

template<size_t N>
double benchmark_static(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t heads,
                        uint32_t ffn_mult, float lr, MeshDevice& dev) {
    // Create input and target
    auto x = traced::make_full(ttnn::Shape({batch, seq, dim}), 1.0f, dev);
    auto target = traced::make_full(ttnn::Shape({batch, seq, dim}), 0.5f, dev);

    // Create model
    traced::TracedTransformerStack<N> model(batch, seq, dim, heads, ffn_mult, lr, dev);

    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        model.train_step(x, target);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; ++i) {
        model.train_step(x, target);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
}

template<size_t N>
double benchmark_traced(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t heads,
                        uint32_t ffn_mult, float lr, MeshDevice& dev) {
    // Create input and target
    auto x = traced::make_full(ttnn::Shape({batch, seq, dim}), 1.0f, dev);
    auto target = traced::make_full(ttnn::Shape({batch, seq, dim}), 0.5f, dev);

    // Create model
    traced::TracedTransformerStack<N> model(batch, seq, dim, heads, ffn_mult, lr, dev);

    // Non-traced warmup first (compile kernels)
    for (int i = 0; i < 2; ++i) {
        model.train_step(x, target);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Capture trace
    auto trace_id = ttnn::operations::trace::begin_trace_capture(&dev, std::nullopt);
    model.train_step(x, target);
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
    // Configuration from environment (NanoGPT defaults)
    uint32_t batch = 32;
    uint32_t seq = 256;
    uint32_t dim = 384;
    uint32_t heads = 6;
    uint32_t ffn_mult = 4;
    uint32_t num_layers = 6;
    float lr = 0.01f;
    size_t trace_mb = 128;   // Standard trace region (128 MB)

    if (const char* env = std::getenv("BENCH_BATCH")) batch = std::stoul(env);
    if (const char* env = std::getenv("BENCH_SEQ")) seq = std::stoul(env);
    if (const char* env = std::getenv("BENCH_DIM")) dim = std::stoul(env);
    if (const char* env = std::getenv("BENCH_HEADS")) heads = std::stoul(env);
    if (const char* env = std::getenv("BENCH_FFN_MULT")) ffn_mult = std::stoul(env);
    if (const char* env = std::getenv("BENCH_LAYERS")) num_layers = std::stoul(env);
    if (const char* env = std::getenv("BENCH_TRACE_MB")) trace_mb = std::stoul(env);

    fmt::print("# Transformer Training Benchmark\n");
    fmt::print("# Config: batch={}, seq={}, dim={}, heads={}, ffn_mult={}, layers={}\n",
               batch, seq, dim, heads, ffn_mult, num_layers);
    fmt::print("# Trace region: {} MB, Warmup: {}, Iterations: {}\n\n",
               trace_mb, N_WARMUP, N_ITERS);

    // Validate dimensions
    if (dim % heads != 0) {
        fmt::print("ERROR: dim ({}) must be divisible by heads ({})\n", dim, heads);
        return 1;
    }
    if (batch % 32 != 0 || seq % 32 != 0 || dim % 32 != 0) {
        fmt::print("ERROR: batch, seq, dim must be multiples of 32\n");
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
            case 1: static_ms = benchmark_static<1>(batch, seq, dim, heads, ffn_mult, lr, device); break;
            case 2: static_ms = benchmark_static<2>(batch, seq, dim, heads, ffn_mult, lr, device); break;
            case 3: static_ms = benchmark_static<3>(batch, seq, dim, heads, ffn_mult, lr, device); break;
            case 6: static_ms = benchmark_static<6>(batch, seq, dim, heads, ffn_mult, lr, device); break;
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
            case 1: traced_ms = benchmark_traced<1>(batch, seq, dim, heads, ffn_mult, lr, device); break;
            case 2: traced_ms = benchmark_traced<2>(batch, seq, dim, heads, ffn_mult, lr, device); break;
            case 3: traced_ms = benchmark_traced<3>(batch, seq, dim, heads, ffn_mult, lr, device); break;
            case 6: traced_ms = benchmark_traced<6>(batch, seq, dim, heads, ffn_mult, lr, device); break;
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
    fmt::print("| Mode     | Total (ms) | Per-layer (ms) | vs Static |\n");
    fmt::print("|----------|------------|----------------|----------|\n");
    fmt::print("| Static   | {:10.3f} | {:14.3f} | {:8.2f}x |\n",
               static_ms, static_ms / num_layers, 1.0);
    if (trace_success) {
        fmt::print("| Traced   | {:10.3f} | {:14.3f} | {:8.2f}x |\n",
                   traced_ms, traced_ms / num_layers, static_ms / traced_ms);
    } else {
        fmt::print("| Traced   |       FAIL |           FAIL |     FAIL |\n");
    }

    // Comparison with TTML
    fmt::print("\n## TTML Comparison (estimated)\n");
    fmt::print("TTML nano_gpt: ~510 ms/step (includes embedding, LayerNorm, GELU)\n");
    fmt::print("Our training:  {:.3f} ms/step (LN + Attention + GELU FFN, no embedding)\n",
               trace_success ? traced_ms : static_ms);
    if (trace_success && traced_ms > 0) {
        fmt::print("Ratio: TTML is {:.2f}x slower (embedding overhead)\n",
                   510.0 / traced_ms);
    }

    fmt::print("\n## Configuration\n");
    fmt::print("batch={}, seq={}, dim={}, heads={}, ffn_mult={}, layers={}\n",
               batch, seq, dim, heads, ffn_mult, num_layers);

    // CSV output
    fmt::print("\n# CSV: batch,seq,dim,heads,layers,static_ms,traced_ms,speedup\n");
    fmt::print("# {},{},{},{},{},{:.3f},{:.3f},{:.2f}\n",
               batch, seq, dim, heads, num_layers,
               static_ms, trace_success ? traced_ms : 0.0,
               trace_success ? static_ms / traced_ms : 0.0);

    return trace_success ? 0 : 1;
}
