// Linear (fused matmul+bias) benchmark - BFP8 vs BF16
#include <ttnn/device.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>
#include <chrono>

using Tensor = tt::tt_metal::Tensor;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;
using DispatchCoreType = tt::tt_metal::DispatchCoreType;

int main() {
    auto device = MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 128*1024*1024, 1,
                                                DispatchCoreConfig{DispatchCoreType::ETH});
    auto& dev = *device;

    uint32_t B = 32, S = 256, D = 512, FFN = 2048;
    int warmup = 10, iters = 100;

    // BF16 tensors
    auto x_bf16 = ttnn::zeros(ttnn::Shape({B, S, D}), ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, dev);
    auto w_bf16 = ttnn::zeros(ttnn::Shape({FFN, D}), ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, dev);
    auto b_bf16 = ttnn::zeros(ttnn::Shape({1, 1, FFN}), ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, dev);

    // BFP8 tensors (including bias!)
    auto x_bfp8 = ttnn::zeros(ttnn::Shape({B, S, D}), ttnn::DataType::BFLOAT8_B, ttnn::TILE_LAYOUT, dev);
    auto w_bfp8 = ttnn::zeros(ttnn::Shape({FFN, D}), ttnn::DataType::BFLOAT8_B, ttnn::TILE_LAYOUT, dev);
    auto b_bfp8 = ttnn::zeros(ttnn::Shape({1, 1, FFN}), ttnn::DataType::BFLOAT8_B, ttnn::TILE_LAYOUT, dev);

    auto hifi2_config = ttnn::WormholeComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi2,
        .math_approx_mode = false,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = true,
    };

    fmt::print("Linear Benchmark: [{}, {}, {}] x [{}, {}] + bias\n", B, S, D, FFN, D);

    // BF16 linear (fused)
    for (int i = 0; i < warmup; i++) {
        ttnn::linear(x_bf16, w_bf16, b_bf16, false, true);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        ttnn::linear(x_bf16, w_bf16, b_bf16, false, true);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto t1 = std::chrono::high_resolution_clock::now();
    double bf16_linear = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // BFP8 linear (fused) - all BFP8
    for (int i = 0; i < warmup; i++) {
        ttnn::linear(x_bfp8, w_bfp8, b_bfp8, false, true,
                     std::nullopt, std::nullopt, std::nullopt, std::nullopt, hifi2_config);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        ttnn::linear(x_bfp8, w_bfp8, b_bfp8, false, true,
                     std::nullopt, std::nullopt, std::nullopt, std::nullopt, hifi2_config);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t1 = std::chrono::high_resolution_clock::now();
    double bfp8_linear = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // BFP8 matmul only (no bias) for reference
    for (int i = 0; i < warmup; i++) {
        ttnn::matmul(x_bfp8, w_bfp8, false, true, std::nullopt,
                     ttnn::DataType::BFLOAT8_B, std::nullopt, std::nullopt, hifi2_config);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        ttnn::matmul(x_bfp8, w_bfp8, false, true, std::nullopt,
                     ttnn::DataType::BFLOAT8_B, std::nullopt, std::nullopt, hifi2_config);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t1 = std::chrono::high_resolution_clock::now();
    double bfp8_matmul = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    fmt::print("\nOperation           | Time (ms) | vs BF16\n");
    fmt::print("--------------------|-----------|--------\n");
    fmt::print("BF16 linear (fused) | {:9.3f} |  1.00x\n", bf16_linear);
    fmt::print("BFP8 linear (fused) | {:9.3f} | {:5.2f}x\n", bfp8_linear, bf16_linear / bfp8_linear);
    fmt::print("BFP8 matmul (no bias)| {:9.3f} | {:5.2f}x\n", bfp8_matmul, bf16_linear / bfp8_matmul);

    tt::tt_metal::distributed::Finish(device->mesh_command_queue());
    ttnn::close_device(*device);
    return 0;
}
