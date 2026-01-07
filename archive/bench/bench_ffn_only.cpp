// Quick FFN-only benchmark to isolate BFP8 vs BF16 performance
#include "static/nn.hpp"
#include <ttnn/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>
#include <chrono>

using namespace static_autograd;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;
using DispatchCoreType = tt::tt_metal::DispatchCoreType;

int main() {
    auto device = MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 128*1024*1024, 1,
                                                DispatchCoreConfig{DispatchCoreType::ETH});
    auto& dev = *device;

    uint32_t batch = 32, seq = 256, dim = 512, ffn_dim = 2048;
    int warmup = 5, iters = 50;

    // Create BF16 input (for BF16 FFN test)
    auto input_bf16 = make_zeros(ttnn::Shape({batch, seq, dim}), dev);

    // Create BFP8 input (for BFP8 FFN test - simulating LayerNormBFP8 output)
    auto input_bfp8 = make_zeros_bfp8(ttnn::Shape({batch, seq, dim}), dev);

    // BF16 FFN
    FFN ffn_bf16(batch, seq, dim, ffn_dim, 0.02f, dev);
    Graph g_bf16;

    // Warmup BF16
    for (int i = 0; i < warmup; i++) {
        auto* inp = g_bf16.leaf(&input_bf16, nullptr, false);
        ffn_bf16.forward(g_bf16, inp);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Time BF16
    auto start_bf16 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        Graph g;
        auto* inp = g.leaf(&input_bf16, nullptr, false);
        ffn_bf16.forward(g, inp);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end_bf16 = std::chrono::high_resolution_clock::now();
    double bf16_ms = std::chrono::duration<double, std::milli>(end_bf16 - start_bf16).count() / iters;

    // BFP8 FFN (expects BFP8 input)
    FFNBFP8 ffn_bfp8(batch, seq, dim, ffn_dim, 0.02f, dev);
    Graph g_bfp8;

    // Warmup BFP8
    for (int i = 0; i < warmup; i++) {
        auto* inp = g_bfp8.leaf(&input_bfp8, nullptr, false);
        ffn_bfp8.forward(g_bfp8, inp);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Time BFP8
    auto start_bfp8 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        Graph g;
        auto* inp = g.leaf(&input_bfp8, nullptr, false);
        ffn_bfp8.forward(g, inp);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end_bfp8 = std::chrono::high_resolution_clock::now();
    double bfp8_ms = std::chrono::duration<double, std::milli>(end_bfp8 - start_bfp8).count() / iters;

    fmt::print("FFN Forward-Only Benchmark (batch={}, seq={}, dim={}, ffn={})\n", batch, seq, dim, ffn_dim);
    fmt::print("BF16:  {:.3f} ms\n", bf16_ms);
    fmt::print("BFP8:  {:.3f} ms (with BFP8 input, no typecast)\n", bfp8_ms);
    fmt::print("Speedup: {:.2f}x\n", bf16_ms / bfp8_ms);

    tt::tt_metal::distributed::Finish(device->mesh_command_queue());
    ttnn::close_device(*device);
    return 0;
}
