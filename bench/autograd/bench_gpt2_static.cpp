// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GPT-2 Training Benchmark using static_autograd
// Tests PersistentGPT2 with trace API to verify tracing speedup.
//
// This benchmark uses the static_autograd pattern where:
// 1. Graph is built once with Tensor* pointers to class members
// 2. Forward execution overwrites same member buffers each iteration
// 3. Trace API captures and replays the execution
//
// Usage:
//   make bench-gpt2-static
//   BENCH_LAYERS=6 make bench-gpt2-static-run

#include "static/nn.hpp"

#include <ttnn/device.hpp>
#include <ttnn/operations/trace.hpp>
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

    DeviceGuard(size_t trace_region_mb)
        : device([trace_region_mb]() {
            bool use_worker = std::getenv("USE_WORKER_DISPATCH") != nullptr;
            return MeshDevice::create_unit_mesh(
                0,
                DEFAULT_L1_SMALL_SIZE,
                trace_region_mb * 1024 * 1024,
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

template<size_t N>
double benchmark_static(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t heads,
                        uint32_t ffn_mult, uint32_t vocab, float lr, MeshDevice& dev) {
    // Create model
    PersistentGPT2<N> model(batch, seq, dim, heads, ffn_mult, vocab, lr, dev);

    // Build graph (this also runs first forward)
    model.build();
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
}

template<size_t N>
double benchmark_static_bfp8(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t heads,
                             uint32_t ffn_mult, uint32_t vocab, float lr, MeshDevice& dev) {
    // Create BFP8 model
    PersistentGPT2BFP8<N> model(batch, seq, dim, heads, ffn_mult, vocab, lr, dev);

    // Build graph (this also runs first forward)
    model.build();
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Warmup
    for (int i = 0; i < N_WARMUP; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
}

template<size_t N>
double benchmark_traced(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t heads,
                        uint32_t ffn_mult, uint32_t vocab, float lr, MeshDevice& dev) {
    // Create model
    PersistentGPT2<N> model(batch, seq, dim, heads, ffn_mult, vocab, lr, dev);

    // Build graph
    model.build();

    // Non-traced warmup (compile kernels)
    for (int i = 0; i < 2; ++i) {
        model.train_step();
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Capture trace
    auto trace_id = ttnn::operations::trace::begin_trace_capture(&dev, std::nullopt);
    model.train_step();
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
    // Configuration from environment
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

    fmt::print("# GPT-2 Training: Static Autograd with Trace API\n");
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
    DeviceGuard dg(trace_mb);
    auto& device = dg.get();
    fmt::print("Device ready\n\n");

    double static_ms = 0.0, static_bfp8_ms = 0.0, traced_ms = 0.0;
    bool bfp8_success = false, trace_success = false;

    // Test 1: Static BF16
    fmt::print("## Test 1: Static BF16 (PersistentGPT2, {} layers)\n", num_layers);
    try {
        switch (num_layers) {
            case 1: static_ms = benchmark_static<1>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 2: static_ms = benchmark_static<2>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 3: static_ms = benchmark_static<3>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 6: static_ms = benchmark_static<6>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
        }
        fmt::print("Static BF16: {:.3f} ms/step, {:.3f} ms/layer\n\n", static_ms, static_ms / num_layers);
    } catch (const std::exception& e) {
        fmt::print("Static BF16 FAILED: {}\n\n", e.what());
        return 1;
    }

    // Test 2: Static BFP8
    fmt::print("## Test 2: Static BFP8 (PersistentGPT2BFP8, {} layers)\n", num_layers);
    try {
        switch (num_layers) {
            case 1: static_bfp8_ms = benchmark_static_bfp8<1>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 2: static_bfp8_ms = benchmark_static_bfp8<2>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 3: static_bfp8_ms = benchmark_static_bfp8<3>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 6: static_bfp8_ms = benchmark_static_bfp8<6>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
        }
        bfp8_success = true;
        fmt::print("Static BFP8: {:.3f} ms/step, {:.3f} ms/layer\n\n", static_bfp8_ms, static_bfp8_ms / num_layers);
    } catch (const std::exception& e) {
        fmt::print("Static BFP8 FAILED: {}\n\n", e.what());
    }

    // Test 3: Traced
    fmt::print("## Test 3: Traced (PersistentGPT2 + trace API, {} layers)\n", num_layers);
    try {
        switch (num_layers) {
            case 1: traced_ms = benchmark_traced<1>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 2: traced_ms = benchmark_traced<2>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 3: traced_ms = benchmark_traced<3>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 6: traced_ms = benchmark_traced<6>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
        }
        trace_success = true;
        fmt::print("Traced: {:.3f} ms/step, {:.3f} ms/layer\n\n", traced_ms, traced_ms / num_layers);
    } catch (const std::exception& e) {
        fmt::print("Traced FAILED: {}\n\n", e.what());
    }

    // Summary
    fmt::print("## Summary\n");
    fmt::print("| Mode         | Total (ms) | Per-layer (ms) | vs BF16 |\n");
    fmt::print("|--------------|------------|----------------|--------|\n");
    fmt::print("| Static BF16  | {:10.3f} | {:14.3f} | {:6.2f}x |\n",
               static_ms, static_ms / num_layers, 1.0);
    if (bfp8_success) {
        double bfp8_speedup = static_ms / static_bfp8_ms;
        fmt::print("| Static BFP8  | {:10.3f} | {:14.3f} | {:6.2f}x |\n",
                   static_bfp8_ms, static_bfp8_ms / num_layers, bfp8_speedup);
    } else {
        fmt::print("| Static BFP8  |       FAIL |           FAIL |   FAIL |\n");
    }
    if (trace_success) {
        double trace_speedup = static_ms / traced_ms;
        fmt::print("| Traced BF16  | {:10.3f} | {:14.3f} | {:6.2f}x |\n",
                   traced_ms, traced_ms / num_layers, trace_speedup);
    } else {
        fmt::print("| Traced BF16  |       FAIL |           FAIL |   FAIL |\n");
    }

    // Analysis
    fmt::print("\n## Analysis\n");
    if (bfp8_success) {
        double bfp8_speedup = static_ms / static_bfp8_ms;
        if (bfp8_speedup > 1.5) {
            fmt::print("✓ BFP8 FFN provides {:.1f}x speedup over BF16!\n", bfp8_speedup);
        } else if (bfp8_speedup > 1.0) {
            fmt::print("△ BFP8 FFN provides {:.1f}x speedup (expected >2x from reduced bandwidth)\n", bfp8_speedup);
        } else {
            fmt::print("✗ BFP8 FFN provides NO speedup ({:.2f}x) - check compute config\n", bfp8_speedup);
        }
    }
    if (trace_success) {
        double trace_speedup = static_ms / traced_ms;
        if (trace_speedup > 1.5) {
            fmt::print("✓ Tracing provides {:.1f}x speedup!\n", trace_speedup);
        } else if (trace_speedup > 1.0) {
            fmt::print("△ Tracing provides {:.1f}x speedup (expected >3x)\n", trace_speedup);
        } else {
            fmt::print("✗ Tracing provides NO speedup ({:.2f}x)\n", trace_speedup);
        }
    }

    // CSV output
    fmt::print("\n# CSV: batch,seq,dim,heads,vocab,layers,bf16_ms,bfp8_ms,bfp8_speedup,traced_ms,trace_speedup\n");
    fmt::print("# {},{},{},{},{},{},{:.3f},{},{},{},{}\n",
               batch, seq, dim, heads, vocab, num_layers,
               static_ms,
               bfp8_success ? fmt::format("{:.3f}", static_bfp8_ms) : "FAIL",
               bfp8_success ? fmt::format("{:.2f}", static_ms / static_bfp8_ms) : "FAIL",
               trace_success ? fmt::format("{:.3f}", traced_ms) : "FAIL",
               trace_success ? fmt::format("{:.2f}", static_ms / traced_ms) : "FAIL");

    return (bfp8_success || trace_success) ? 0 : 1;
}
