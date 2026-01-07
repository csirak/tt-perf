// Add benchmark - same type vs mixed type
#include <ttnn/device.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
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

    uint32_t B = 32, S = 256, FFN = 2048;
    int warmup = 10, iters = 100;

    // Tensors
    auto x_bf16 = ttnn::zeros(ttnn::Shape({B, S, FFN}), ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, dev);
    auto b_bf16 = ttnn::zeros(ttnn::Shape({1, 1, FFN}), ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, dev);
    auto x_bfp8 = ttnn::zeros(ttnn::Shape({B, S, FFN}), ttnn::DataType::BFLOAT8_B, ttnn::TILE_LAYOUT, dev);
    auto b_bfp8 = ttnn::zeros(ttnn::Shape({1, 1, FFN}), ttnn::DataType::BFLOAT8_B, ttnn::TILE_LAYOUT, dev);

    fmt::print("Add Benchmark: [{}, {}, {}] + [{}, {}, {}]\n", B, S, FFN, 1, 1, FFN);

    // BF16 + BF16
    for (int i = 0; i < warmup; i++) ttnn::add(x_bf16, b_bf16);
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) ttnn::add(x_bf16, b_bf16);
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    auto t1 = std::chrono::high_resolution_clock::now();
    double bf16_bf16 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // BFP8 + BFP8
    for (int i = 0; i < warmup; i++) ttnn::add(x_bfp8, b_bfp8);
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) ttnn::add(x_bfp8, b_bfp8);
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t1 = std::chrono::high_resolution_clock::now();
    double bfp8_bfp8 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // BFP8 + BF16 (mixed - current code)
    for (int i = 0; i < warmup; i++) ttnn::add(x_bfp8, b_bf16);
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) ttnn::add(x_bfp8, b_bf16);
    tt::tt_metal::distributed::Synchronize(&dev, std::nullopt);
    t1 = std::chrono::high_resolution_clock::now();
    double bfp8_bf16 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    fmt::print("\nOperation      | Time (ms)\n");
    fmt::print("---------------|----------\n");
    fmt::print("BF16 + BF16    | {:9.3f}\n", bf16_bf16);
    fmt::print("BFP8 + BFP8    | {:9.3f}\n", bfp8_bfp8);
    fmt::print("BFP8 + BF16    | {:9.3f}  <-- PROBLEM!\n", bfp8_bf16);

    tt::tt_metal::distributed::Finish(device->mesh_command_queue());
    ttnn::close_device(*device);
    return 0;
}
