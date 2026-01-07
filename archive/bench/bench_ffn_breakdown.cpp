// FFN breakdown benchmark - identify where BFP8 slowdown occurs
#include <ttnn/device.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>
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
    auto w1_bf16 = ttnn::zeros(ttnn::Shape({FFN, D}), ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, dev);
    auto w2_bf16 = ttnn::zeros(ttnn::Shape({D, FFN}), ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, dev);
    auto b1_bf16 = ttnn::zeros(ttnn::Shape({1, 1, FFN}), ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, dev);
    auto b2_bf16 = ttnn::zeros(ttnn::Shape({1, 1, D}), ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, dev);

    // BFP8 tensors
    auto x_bfp8 = ttnn::zeros(ttnn::Shape({B, S, D}), ttnn::DataType::BFLOAT8_B, ttnn::TILE_LAYOUT, dev);
    auto w1_bfp8 = ttnn::zeros(ttnn::Shape({FFN, D}), ttnn::DataType::BFLOAT8_B, ttnn::TILE_LAYOUT, dev);
    auto w2_bfp8 = ttnn::zeros(ttnn::Shape({D, FFN}), ttnn::DataType::BFLOAT8_B, ttnn::TILE_LAYOUT, dev);

    auto hifi2_config = ttnn::WormholeComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi2,
        .math_approx_mode = false,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = true,
    };

    fmt::print("FFN Breakdown: [{}, {}, {}] -> [{}, {}, {}] -> [{}, {}, {}]\n", B, S, D, B, S, FFN, B, S, D);

    // ===== BF16 Matmul 1 =====
    for (int i = 0; i < warmup; i++) ttnn::matmul(x_bf16, w1_bf16, false, true);
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) ttnn::matmul(x_bf16, w1_bf16, false, true);
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto t1 = std::chrono::high_resolution_clock::now();
    double bf16_mm1 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // ===== BF16 Matmul 1 + Add =====
    for (int i = 0; i < warmup; i++) {
        auto mm = ttnn::matmul(x_bf16, w1_bf16, false, true);
        ttnn::add(mm, b1_bf16);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        auto mm = ttnn::matmul(x_bf16, w1_bf16, false, true);
        ttnn::add(mm, b1_bf16);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t1 = std::chrono::high_resolution_clock::now();
    double bf16_mm1_add = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // ===== BF16 Full (Matmul + Add + GELU) =====
    for (int i = 0; i < warmup; i++) {
        auto mm = ttnn::matmul(x_bf16, w1_bf16, false, true);
        auto added = ttnn::add(mm, b1_bf16);
        ttnn::gelu(added, true);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        auto mm = ttnn::matmul(x_bf16, w1_bf16, false, true);
        auto added = ttnn::add(mm, b1_bf16);
        ttnn::gelu(added, true);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t1 = std::chrono::high_resolution_clock::now();
    double bf16_full1 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // ===== BFP8 Matmul 1 =====
    for (int i = 0; i < warmup; i++) {
        ttnn::matmul(x_bfp8, w1_bfp8, false, true, std::nullopt,
                     ttnn::DataType::BFLOAT8_B, std::nullopt, std::nullopt, hifi2_config);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        ttnn::matmul(x_bfp8, w1_bfp8, false, true, std::nullopt,
                     ttnn::DataType::BFLOAT8_B, std::nullopt, std::nullopt, hifi2_config);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t1 = std::chrono::high_resolution_clock::now();
    double bfp8_mm1 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // ===== BFP8 Matmul 1 + Add =====
    for (int i = 0; i < warmup; i++) {
        auto mm = ttnn::matmul(x_bfp8, w1_bfp8, false, true, std::nullopt,
                               ttnn::DataType::BFLOAT8_B, std::nullopt, std::nullopt, hifi2_config);
        ttnn::add(mm, b1_bf16);  // Mixed: BFP8 + BF16 bias
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        auto mm = ttnn::matmul(x_bfp8, w1_bfp8, false, true, std::nullopt,
                               ttnn::DataType::BFLOAT8_B, std::nullopt, std::nullopt, hifi2_config);
        ttnn::add(mm, b1_bf16);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t1 = std::chrono::high_resolution_clock::now();
    double bfp8_mm1_add = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // ===== BFP8 Full (Matmul + Add + GELU) =====
    for (int i = 0; i < warmup; i++) {
        auto mm = ttnn::matmul(x_bfp8, w1_bfp8, false, true, std::nullopt,
                               ttnn::DataType::BFLOAT8_B, std::nullopt, std::nullopt, hifi2_config);
        auto added = ttnn::add(mm, b1_bf16);
        ttnn::gelu(added, true);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        auto mm = ttnn::matmul(x_bfp8, w1_bfp8, false, true, std::nullopt,
                               ttnn::DataType::BFLOAT8_B, std::nullopt, std::nullopt, hifi2_config);
        auto added = ttnn::add(mm, b1_bf16);
        ttnn::gelu(added, true);
    }
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t1 = std::chrono::high_resolution_clock::now();
    double bfp8_full1 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    fmt::print("\n=== Layer 1 (up-projection) ===\n");
    fmt::print("Operation          | BF16 (ms) | BFP8 (ms) | Speedup\n");
    fmt::print("-------------------|-----------|-----------|--------\n");
    fmt::print("Matmul only        | {:9.3f} | {:9.3f} | {:6.2f}x\n", bf16_mm1, bfp8_mm1, bf16_mm1 / bfp8_mm1);
    fmt::print("Matmul + Add       | {:9.3f} | {:9.3f} | {:6.2f}x\n", bf16_mm1_add, bfp8_mm1_add, bf16_mm1_add / bfp8_mm1_add);
    fmt::print("Matmul + Add + GELU| {:9.3f} | {:9.3f} | {:6.2f}x\n", bf16_full1, bfp8_full1, bf16_full1 / bfp8_full1);

    fmt::print("\n=== Deltas ===\n");
    fmt::print("Add overhead:  BF16={:.3f}ms, BFP8={:.3f}ms\n", bf16_mm1_add - bf16_mm1, bfp8_mm1_add - bfp8_mm1);
    fmt::print("GELU overhead: BF16={:.3f}ms, BFP8={:.3f}ms\n", bf16_full1 - bf16_mm1_add, bfp8_full1 - bfp8_mm1_add);

    tt::tt_metal::distributed::Finish(device->mesh_command_queue());
    ttnn::close_device(*device);
    return 0;
}
