// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GPT-2 Forward+Backward Benchmark (BF16 only)
// Simple benchmark to measure single step time without SGD overhead.
//
// Usage:
//   make bench-gpt2-fwd-back
//   BENCH_LAYERS=6 make bench-gpt2-fwd-back-run

#include "static/nn.hpp"

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

template<size_t N>
double benchmark_fwd_back(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t heads,
                          uint32_t ffn_mult, uint32_t vocab, float lr, MeshDevice& dev) {
    // Create BF16 model
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

    fmt::print("# GPT-2 Forward+Backward Benchmark (BF16)\n");
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

    double ms_per_step = 0.0;

    fmt::print("## BF16 Forward+Backward ({} layers)\n", num_layers);
    try {
        switch (num_layers) {
            case 1: ms_per_step = benchmark_fwd_back<1>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 2: ms_per_step = benchmark_fwd_back<2>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 3: ms_per_step = benchmark_fwd_back<3>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
            case 6: ms_per_step = benchmark_fwd_back<6>(batch, seq, dim, heads, ffn_mult, vocab, lr, device); break;
        }
        fmt::print("BF16: {:.3f} ms/step, {:.3f} ms/layer\n\n", ms_per_step, ms_per_step / num_layers);
    } catch (const std::exception& e) {
        fmt::print("FAILED: {}\n\n", e.what());
        return 1;
    }

    // Calculate FLOPS
    uint32_t ffn_dim = dim * ffn_mult;
    uint64_t head_dim = dim / heads;

    // Per transformer layer FLOPS (forward + backward = 3x forward)
    // QKV: 3 * 2 * batch * seq * dim * dim = 6 * B * S * D^2
    // Attention scores: 2 * batch * heads * seq * seq * head_dim = 2 * B * H * S^2 * d
    // Attention output: 2 * batch * heads * seq * head_dim * seq = 2 * B * H * S * d * S
    // Output proj: 2 * batch * seq * dim * dim = 2 * B * S * D^2
    // FFN up: 2 * batch * seq * dim * ffn_dim = 2 * B * S * D * F
    // FFN down: 2 * batch * seq * ffn_dim * dim = 2 * B * S * F * D
    // Total forward per layer: 8*B*S*D^2 + 4*B*H*S^2*d + 4*B*S*D*F
    // Total fwd+bwd: 3x forward (for gradients)

    uint64_t B = batch, S = seq, D = dim, H = heads, d = head_dim, F = ffn_dim;
    uint64_t layer_flops_fwd = 8*B*S*D*D + 4*B*H*S*S*d + 4*B*S*D*F;
    uint64_t layer_flops_total = 3 * layer_flops_fwd;  // fwd + bwd (2x fwd for grads)
    uint64_t model_flops = num_layers * layer_flops_total;

    // Output projection
    model_flops += 3 * 2 * B * S * D * vocab;

    double tflops = (model_flops / 1e12) / (ms_per_step / 1000.0);
    double utilization = tflops / 70.0 * 100.0;  // Peak BF16 HiFi2 = 70 TFLOPS

    fmt::print("## Performance Analysis\n");
    fmt::print("| Metric | Value |\n");
    fmt::print("|--------|-------|\n");
    fmt::print("| Total FLOPS | {:.2f} GFLOPS |\n", model_flops / 1e9);
    fmt::print("| Achieved | {:.2f} TFLOPS |\n", tflops);
    fmt::print("| Peak (BF16 HiFi2) | 70 TFLOPS |\n");
    fmt::print("| Utilization | {:.1f}% |\n", utilization);

    fmt::print("\n# CSV: batch,seq,dim,heads,vocab,layers,ms_per_step,tflops,utilization\n");
    fmt::print("# {},{},{},{},{},{},{:.3f},{:.2f},{:.1f}\n",
               batch, seq, dim, heads, vocab, num_layers, ms_per_step, tflops, utilization);

    return 0;
}
