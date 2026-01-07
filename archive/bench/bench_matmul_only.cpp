// Minimal matmul benchmark - BFP8 vs BF16
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

    // FFN matmul shapes: [32, 256, 512] x [2048, 512] = [32, 256, 2048]
    uint32_t B = 32, S = 256, D = 512, FFN = 2048;
    int warmup = 10, iters = 100;

    // BF16 tensors
    auto x_bf16 = ttnn::zeros(ttnn::Shape({B, S, D}), ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, dev);
    auto w_bf16 = ttnn::zeros(ttnn::Shape({FFN, D}), ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, dev);

    // BFP8 tensors
    auto x_bfp8 = ttnn::zeros(ttnn::Shape({B, S, D}), ttnn::DataType::BFLOAT8_B, ttnn::TILE_LAYOUT, dev);
    auto w_bfp8 = ttnn::zeros(ttnn::Shape({FFN, D}), ttnn::DataType::BFLOAT8_B, ttnn::TILE_LAYOUT, dev);

    // Compute configs - use MathFidelity directly (no namespace needed)
    auto hifi2_config = ttnn::WormholeComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi2,
        .math_approx_mode = false,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = true,
    };

    auto lofi_config = ttnn::WormholeComputeKernelConfig{
        .math_fidelity = MathFidelity::LoFi,
        .math_approx_mode = true,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = true,
    };

    // Warmup BF16
    for (int i = 0; i < warmup; i++) {
        auto y = ttnn::matmul(x_bf16, w_bf16, false, true);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Time BF16
    auto start_bf16 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        auto y = ttnn::matmul(x_bf16, w_bf16, false, true);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end_bf16 = std::chrono::high_resolution_clock::now();
    double bf16_ms = std::chrono::duration<double, std::milli>(end_bf16 - start_bf16).count() / iters;

    // Warmup BFP8 HiFi2
    for (int i = 0; i < warmup; i++) {
        auto y = ttnn::matmul(x_bfp8, w_bfp8, false, true, std::nullopt,
                              ttnn::DataType::BFLOAT8_B, std::nullopt, std::nullopt, hifi2_config);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Time BFP8 HiFi2
    auto start_bfp8_hifi2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        auto y = ttnn::matmul(x_bfp8, w_bfp8, false, true, std::nullopt,
                              ttnn::DataType::BFLOAT8_B, std::nullopt, std::nullopt, hifi2_config);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end_bfp8_hifi2 = std::chrono::high_resolution_clock::now();
    double bfp8_hifi2_ms = std::chrono::duration<double, std::milli>(end_bfp8_hifi2 - start_bfp8_hifi2).count() / iters;

    // Warmup BFP8 LoFi
    for (int i = 0; i < warmup; i++) {
        auto y = ttnn::matmul(x_bfp8, w_bfp8, false, true, std::nullopt,
                              ttnn::DataType::BFLOAT8_B, std::nullopt, std::nullopt, lofi_config);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);

    // Time BFP8 LoFi
    auto start_bfp8_lofi = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        auto y = ttnn::matmul(x_bfp8, w_bfp8, false, true, std::nullopt,
                              ttnn::DataType::BFLOAT8_B, std::nullopt, std::nullopt, lofi_config);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto end_bfp8_lofi = std::chrono::high_resolution_clock::now();
    double bfp8_lofi_ms = std::chrono::duration<double, std::milli>(end_bfp8_lofi - start_bfp8_lofi).count() / iters;

    fmt::print("Matmul Benchmark: [{}, {}, {}] x [{}, {}] (transpose_b=true)\n", B, S, D, FFN, D);
    fmt::print("BF16 (default):  {:.3f} ms\n", bf16_ms);
    fmt::print("BFP8 (HiFi2):    {:.3f} ms  ({:.2f}x vs BF16)\n", bfp8_hifi2_ms, bf16_ms / bfp8_hifi2_ms);
    fmt::print("BFP8 (LoFi):     {:.3f} ms  ({:.2f}x vs BF16)\n", bfp8_lofi_ms, bf16_ms / bfp8_lofi_ms);

    tt::tt_metal::distributed::Finish(device->mesh_command_queue());
    ttnn::close_device(*device);
    return 0;
}
