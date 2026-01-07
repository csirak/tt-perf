// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Transformer Trace Benchmark: Compare trace patterns
//
// Tests 3 execution modes (all use pre-allocated TracedTransformerLayer):
//   1. Static (control) - forward() N times, no trace
//   2. Trace 1 + replay N - trace 1 forward, execute_trace N times
//   3. Trace N + replay 1 - trace N forwards, execute_trace once
//
// Usage:
//   make bench-transformer-depth
//   BENCH_BATCH=32 BENCH_SEQ=128 BENCH_DIM=256 BENCH_LAYERS=5 make bench-transformer-depth-run

#include "traced/transformer.hpp"
#include "traced/trace.hpp"
#include "common.hpp"

#include <ttnn/operations/trace.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>
#include <chrono>
#include <cstdlib>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;

// Custom DeviceGuard with configurable trace region size
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

constexpr int N_WARMUP = 5;
constexpr int N_ITERS = 20;

int main() {
    // Configuration from environment
    uint32_t batch = 32, seq = 128, dim = 256, heads = 4, ffn_mult = 4;
    uint32_t num_layers = 5;  // Default to 5 layers
    size_t trace_mb = 128;    // Standard trace region (128 MB)
    const float init_val = 0.01f;

    if (const char* env = std::getenv("BENCH_BATCH")) batch = std::stoul(env);
    if (const char* env = std::getenv("BENCH_SEQ")) seq = std::stoul(env);
    if (const char* env = std::getenv("BENCH_DIM")) dim = std::stoul(env);
    if (const char* env = std::getenv("BENCH_HEADS")) heads = std::stoul(env);
    if (const char* env = std::getenv("BENCH_FFN_MULT")) ffn_mult = std::stoul(env);
    if (const char* env = std::getenv("BENCH_LAYERS")) num_layers = std::stoul(env);
    if (const char* env = std::getenv("BENCH_TRACE_MB")) trace_mb = std::stoul(env);

    fmt::print("# Transformer Trace Benchmark\n");
    fmt::print("# Config: batch={}, seq={}, dim={}, heads={}, ffn_mult={}\n",
               batch, seq, dim, heads, ffn_mult);
    fmt::print("# Layers: {}, Trace region: {} MB\n", num_layers, trace_mb);
    fmt::print("# Warmup: {}, Iterations: {}\n\n", N_WARMUP, N_ITERS);

    // Validate dimensions
    if (dim % heads != 0) {
        fmt::print("ERROR: dim ({}) must be divisible by heads ({})\n", dim, heads);
        return 1;
    }
    if (batch % 32 != 0 || seq % 32 != 0 || dim % 32 != 0) {
        fmt::print("ERROR: batch, seq, dim must be multiples of 32\n");
        return 1;
    }

    // Open device
    fmt::print("Opening device with {} MB trace region...\n", trace_mb);
    ConfigurableDeviceGuard dg(trace_mb);
    auto& device = dg.get();
    fmt::print("Device ready\n\n");

    // Create input tensor and layer (reused across all tests)
    auto input = traced::make_full(ttnn::Shape({batch, seq, dim}), 1.0f, device);
    traced::TracedTransformerLayer layer(batch, seq, dim, heads, ffn_mult, init_val, device);

    double static_ms = 0.0, trace1_replay_n_ms = 0.0, trace_n_replay_1_ms = 0.0;
    bool trace1_success = false, trace_n_success = false;

    // =========================================================================
    // Test 1: Static execution (CONTROL - pre-allocated, no trace)
    // =========================================================================
    fmt::print("## Test 1: Static execution (control)\n");
    {
        // Warmup: compile kernels
        for (int i = 0; i < N_WARMUP; ++i) {
            layer.forward(input);
        }
        tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        fmt::print("Warmup complete\n");

        // Timed runs
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_ITERS; ++i) {
            for (uint32_t l = 0; l < num_layers; ++l) {
                layer.forward(input);
            }
        }
        tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();

        static_ms = std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
        fmt::print("Static: {:.3f} ms total, {:.3f} ms/layer\n\n",
                   static_ms, static_ms / num_layers);
    }

    // =========================================================================
    // Test 2: Trace 1 layer + replay N times
    // =========================================================================
    fmt::print("## Test 2: Trace 1 layer + execute_trace N times\n");
    try {
        // Warmup (compile kernels - already done but be safe)
        for (int i = 0; i < 2; ++i) {
            layer.forward(input);
        }
        tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

        // Capture trace of single forward
        fmt::print("Capturing trace of 1 forward...\n");
        auto trace_id = ttnn::operations::trace::begin_trace_capture(&device, std::nullopt);
        layer.forward(input);
        ttnn::operations::trace::end_trace_capture(&device, trace_id, std::nullopt);
        fmt::print("Trace captured\n");

        // Warmup trace replay
        for (int i = 0; i < N_WARMUP; ++i) {
            for (uint32_t l = 0; l < num_layers; ++l) {
                ttnn::operations::trace::execute_trace(&device, trace_id, std::nullopt, false);
            }
        }
        tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        fmt::print("Trace warmup complete\n");

        // Timed runs
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_ITERS; ++i) {
            for (uint32_t l = 0; l < num_layers; ++l) {
                ttnn::operations::trace::execute_trace(&device, trace_id, std::nullopt, false);
            }
        }
        tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();

        trace1_replay_n_ms = std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
        trace1_success = true;
        fmt::print("Trace 1 + replay N: {:.3f} ms total, {:.3f} ms/layer\n\n",
                   trace1_replay_n_ms, trace1_replay_n_ms / num_layers);

        ttnn::operations::trace::release_trace(&device, trace_id);
    } catch (const std::exception& e) {
        fmt::print("Trace 1 FAILED: {}\n\n", e.what());
    }

    // =========================================================================
    // Test 3: Trace N layers + replay once
    // =========================================================================
    fmt::print("## Test 3: Trace {} layers + execute_trace once\n", num_layers);
    try {
        // Warmup
        for (int i = 0; i < 2; ++i) {
            layer.forward(input);
        }
        tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

        // Capture trace of N forwards
        fmt::print("Capturing trace of {} forwards...\n", num_layers);
        auto trace_id = ttnn::operations::trace::begin_trace_capture(&device, std::nullopt);
        for (uint32_t l = 0; l < num_layers; ++l) {
            layer.forward(input);
        }
        ttnn::operations::trace::end_trace_capture(&device, trace_id, std::nullopt);
        fmt::print("Trace captured\n");

        // Warmup trace replay
        for (int i = 0; i < N_WARMUP; ++i) {
            ttnn::operations::trace::execute_trace(&device, trace_id, std::nullopt, false);
        }
        tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        fmt::print("Trace warmup complete\n");

        // Timed runs
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_ITERS; ++i) {
            ttnn::operations::trace::execute_trace(&device, trace_id, std::nullopt, false);
        }
        tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();

        trace_n_replay_1_ms = std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
        trace_n_success = true;
        fmt::print("Trace N + replay 1: {:.3f} ms total, {:.3f} ms/layer\n\n",
                   trace_n_replay_1_ms, trace_n_replay_1_ms / num_layers);

        ttnn::operations::trace::release_trace(&device, trace_id);
    } catch (const std::exception& e) {
        fmt::print("Trace N FAILED: {}\n\n", e.what());
    }

    // =========================================================================
    // Summary
    // =========================================================================
    fmt::print("## Summary\n");
    fmt::print("| Mode                    | Total (ms) | Per-layer (ms) | vs Static |\n");
    fmt::print("|-------------------------|------------|----------------|----------|\n");
    fmt::print("| Static (control)        | {:10.3f} | {:14.3f} | {:8.2f}x |\n",
               static_ms, static_ms / num_layers, 1.0);
    if (trace1_success) {
        fmt::print("| Trace 1 + replay N      | {:10.3f} | {:14.3f} | {:8.2f}x |\n",
                   trace1_replay_n_ms, trace1_replay_n_ms / num_layers,
                   static_ms / trace1_replay_n_ms);
    } else {
        fmt::print("| Trace 1 + replay N      |       FAIL |           FAIL |     FAIL |\n");
    }
    if (trace_n_success) {
        fmt::print("| Trace N + replay 1      | {:10.3f} | {:14.3f} | {:8.2f}x |\n",
                   trace_n_replay_1_ms, trace_n_replay_1_ms / num_layers,
                   static_ms / trace_n_replay_1_ms);
    } else {
        fmt::print("| Trace N + replay 1      |       FAIL |           FAIL |     FAIL |\n");
    }

    fmt::print("\n## Configuration\n");
    fmt::print("batch={}, seq={}, dim={}, heads={}, ffn_mult={}, layers={}, trace_mb={}\n",
               batch, seq, dim, heads, ffn_mult, num_layers, trace_mb);
    fmt::print("warmup={}, iters={}\n", N_WARMUP, N_ITERS);

    // CSV output
    fmt::print("\n# CSV: batch,seq,dim,heads,layers,static_ms,trace1_ms,traceN_ms,trace1_speedup,traceN_speedup\n");
    fmt::print("# {},{},{},{},{},{:.3f},{:.3f},{:.3f},{:.2f},{:.2f}\n",
               batch, seq, dim, heads, num_layers,
               static_ms,
               trace1_success ? trace1_replay_n_ms : 0.0,
               trace_n_success ? trace_n_replay_1_ms : 0.0,
               trace1_success ? static_ms / trace1_replay_n_ms : 0.0,
               trace_n_success ? static_ms / trace_n_replay_1_ms : 0.0);

    return (trace1_success && trace_n_success) ? 0 : 1;
}
