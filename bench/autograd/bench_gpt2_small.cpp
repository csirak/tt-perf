// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GPT-2 Small Benchmark: 12 stacked transformer layers
//
// GPT-2 Small specs:
//   - 12 layers, 768 dim, 12 heads, 3072 FFN (4x)
//   - Context length: 1024 (configurable via BENCH_SEQ)
//
// Tests:
//   1. Static - forward through 12 layers, no trace
//   2. Trace all 12 layers + replay
//
// Usage:
//   make bench-gpt2-small
//   BENCH_BATCH=32 BENCH_SEQ=512 make bench-gpt2-small-run

#include "traced/transformer.hpp"
#include "traced/trace.hpp"
#include "common.hpp"

#include <ttnn/operations/trace.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>
#include <chrono>
#include <cstdlib>
#include <vector>

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

// GPT-2 Small defaults
constexpr uint32_t GPT2_DIM = 768;
constexpr uint32_t GPT2_HEADS = 12;
constexpr uint32_t GPT2_FFN_MULT = 4;
constexpr uint32_t GPT2_LAYERS = 12;

int main() {
    // Configuration - GPT-2 Small defaults
    uint32_t batch = 32;
    uint32_t seq = 256;  // Use 256 to fit in memory, 512+ causes OOM with 12 layers
    uint32_t dim = GPT2_DIM;
    uint32_t heads = GPT2_HEADS;
    uint32_t ffn_mult = GPT2_FFN_MULT;
    uint32_t num_layers = GPT2_LAYERS;
    size_t trace_mb = 128;   // Standard trace region (128 MB)
    const float init_val = 0.01f;

    if (const char* env = std::getenv("BENCH_BATCH")) batch = std::stoul(env);
    if (const char* env = std::getenv("BENCH_SEQ")) seq = std::stoul(env);
    if (const char* env = std::getenv("BENCH_DIM")) dim = std::stoul(env);
    if (const char* env = std::getenv("BENCH_HEADS")) heads = std::stoul(env);
    if (const char* env = std::getenv("BENCH_FFN_MULT")) ffn_mult = std::stoul(env);
    if (const char* env = std::getenv("BENCH_LAYERS")) num_layers = std::stoul(env);
    if (const char* env = std::getenv("BENCH_TRACE_MB")) trace_mb = std::stoul(env);

    fmt::print("# GPT-2 Small Benchmark ({} layers)\n", num_layers);
    fmt::print("# Config: batch={}, seq={}, dim={}, heads={}, ffn_mult={}\n",
               batch, seq, dim, heads, ffn_mult);
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

    // Open device
    fmt::print("Opening device with {} MB trace region...\n", trace_mb);
    ConfigurableDeviceGuard dg(trace_mb);
    auto& device = dg.get();
    fmt::print("Device ready\n\n");

    // Create input tensor
    fmt::print("Creating input tensor [{}, {}, {}]...\n", batch, seq, dim);
    auto input = traced::make_full(ttnn::Shape({batch, seq, dim}), 1.0f, device);

    // Create 12 transformer layers (each with own weights)
    fmt::print("Creating {} transformer layers...\n", num_layers);
    std::vector<traced::TracedTransformerLayer> layers;
    layers.reserve(num_layers);
    for (uint32_t i = 0; i < num_layers; ++i) {
        layers.emplace_back(batch, seq, dim, heads, ffn_mult, init_val, device);
        if ((i + 1) % 4 == 0) {
            fmt::print("  Created layer {}/{}\n", i + 1, num_layers);
        }
    }
    fmt::print("All layers created\n\n");

    // Lambda to run forward through all layers
    auto forward_all = [&](const Tensor& x) {
        layers[0].forward(x);
        for (uint32_t i = 1; i < num_layers; ++i) {
            layers[i].forward(layers[i - 1].get_output());
        }
    };

    double static_ms = 0.0, traced_ms = 0.0;
    bool trace_success = false;

    // =========================================================================
    // Test 1: Static execution (no trace)
    // =========================================================================
    fmt::print("## Test 1: Static execution (12 layers, no trace)\n");
    {
        // Warmup
        for (int i = 0; i < N_WARMUP; ++i) {
            forward_all(input);
        }
        tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        fmt::print("Warmup complete\n");

        // Timed runs
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_ITERS; ++i) {
            forward_all(input);
        }
        tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();

        static_ms = std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
        fmt::print("Static: {:.3f} ms total, {:.3f} ms/layer\n\n",
                   static_ms, static_ms / num_layers);
    }

    // Verify output
    float out_val = layers.back().get_first_output(&device);
    fmt::print("Output sample (layer {}): {:.6f}\n\n", num_layers, out_val);

    // =========================================================================
    // Test 2: Trace all 12 layers + replay
    // =========================================================================
    fmt::print("## Test 2: Trace {} layers + execute_trace\n", num_layers);
    try {
        // Warmup (compile kernels)
        for (int i = 0; i < 2; ++i) {
            forward_all(input);
        }
        tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

        // Capture trace of full forward
        fmt::print("Capturing trace of {} layers...\n", num_layers);
        auto trace_id = ttnn::operations::trace::begin_trace_capture(&device, std::nullopt);
        forward_all(input);
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

        traced_ms = std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
        trace_success = true;
        fmt::print("Traced: {:.3f} ms total, {:.3f} ms/layer\n\n",
                   traced_ms, traced_ms / num_layers);

        ttnn::operations::trace::release_trace(&device, trace_id);
    } catch (const std::exception& e) {
        fmt::print("Trace FAILED: {}\n\n", e.what());
    }

    // =========================================================================
    // Summary
    // =========================================================================
    fmt::print("## Summary\n");
    fmt::print("| Mode                    | Total (ms) | Per-layer (ms) | vs Static |\n");
    fmt::print("|-------------------------|------------|----------------|----------|\n");
    fmt::print("| Static (control)        | {:10.3f} | {:14.3f} | {:8.2f}x |\n",
               static_ms, static_ms / num_layers, 1.0);
    if (trace_success) {
        fmt::print("| Traced (12 layers)      | {:10.3f} | {:14.3f} | {:8.2f}x |\n",
                   traced_ms, traced_ms / num_layers, static_ms / traced_ms);
    } else {
        fmt::print("| Traced (12 layers)      |       FAIL |           FAIL |     FAIL |\n");
    }

    fmt::print("\n## Configuration\n");
    fmt::print("batch={}, seq={}, dim={}, heads={}, ffn_mult={}, layers={}, trace_mb={}\n",
               batch, seq, dim, heads, ffn_mult, num_layers, trace_mb);

    // CSV output
    fmt::print("\n# CSV: batch,seq,dim,heads,layers,static_ms,traced_ms,speedup\n");
    fmt::print("# {},{},{},{},{},{:.3f},{:.3f},{:.2f}\n",
               batch, seq, dim, heads, num_layers,
               static_ms, trace_success ? traced_ms : 0.0,
               trace_success ? static_ms / traced_ms : 0.0);

    return trace_success ? 0 : 1;
}
