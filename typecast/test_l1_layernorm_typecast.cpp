// Test: In-place L1 Operations Pattern
// Goal: Chain LayerNorm → Typecast entirely in L1 (avoid DRAM round-trip)
//
// Pattern: Use ttnn::L1_MEMORY_CONFIG for both ops
// - LayerNorm outputs to L1
// - Typecast reads from L1, outputs to L1

#include <chrono>
#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <tt-metalium/distributed.hpp>

#include <fmt/core.h>

using namespace ttnn;
using namespace tt::tt_metal::distributed;

constexpr int WARMUP_ITERS = 3;
constexpr int BENCH_ITERS = 10;

void print_memory_config(const Tensor& t, const std::string& name) {
    auto mem_cfg = t.memory_config();
    std::string layout_str = "UNKNOWN";
    if (mem_cfg.memory_layout() == TensorMemoryLayout::INTERLEAVED) {
        layout_str = "INTERLEAVED";
    } else if (mem_cfg.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        layout_str = "HEIGHT_SHARDED";
    } else if (mem_cfg.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        layout_str = "BLOCK_SHARDED";
    } else if (mem_cfg.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        layout_str = "WIDTH_SHARDED";
    }

    std::string buffer_str = (mem_cfg.buffer_type() == BufferType::L1) ? "L1" : "DRAM";

    fmt::print("  {}: {} in {} ({})\n", name, layout_str, buffer_str,
               t.dtype() == DataType::BFLOAT16 ? "BF16" :
               t.dtype() == DataType::BFLOAT8_B ? "BFP8" : "other");
}

int main() {
    fmt::print("=== L1 IN-PLACE OPERATIONS TEST ===\n");
    fmt::print("Goal: Chain LayerNorm → Typecast in L1 (no DRAM round-trip)\n\n");

    // Open device
    using tt::tt_metal::DispatchCoreConfig;
    using tt::tt_metal::DispatchCoreType;
    auto device = MeshDevice::create_unit_mesh(
        0,                           // device_id
        DEFAULT_L1_SMALL_SIZE,       // l1_small_size
        128 * 1024 * 1024,           // trace_region_size
        1,                           // num_command_queues
        DispatchCoreConfig{DispatchCoreType::ETH}
    );

    auto grid = device->compute_with_storage_grid_size();
    fmt::print("Compute grid: {}x{} = {} cores\n\n", grid.x, grid.y, grid.x * grid.y);

    MeshCommandQueue& cq = device->mesh_command_queue();

    // GPT-2 hidden shape: [1, 1, 1024, 768]
    ttnn::Shape shape({1, 1, 1024, 768});
    fmt::print("Test shape: [1,1,1024,768] ({} elements)\n\n", 1024 * 768);

    // Create input tensor in DRAM
    auto input = ones(shape, DataType::BFLOAT16, TILE_LAYOUT, *device);

    // Create weight and bias for LayerNorm (1D tensors for last dim)
    auto weight = ones(ttnn::Shape({1, 1, 32, 768}), DataType::BFLOAT16, TILE_LAYOUT, *device);
    auto bias = zeros(ttnn::Shape({1, 1, 32, 768}), DataType::BFLOAT16, TILE_LAYOUT, *device);
    Finish(cq);

    fmt::print("=== Method 1: DRAM Baseline ===\n");
    fmt::print("LayerNorm(DRAM) → Typecast(DRAM)\n");
    {
        // LayerNorm outputs to DRAM (default)
        auto ln_out = ttnn::layer_norm(input, 1e-5f, weight, bias);
        Finish(cq);
        print_memory_config(ln_out, "ln_out");

        // Typecast outputs to DRAM (default)
        auto result = ttnn::typecast(ln_out, DataType::BFLOAT8_B);
        Finish(cq);
        print_memory_config(result, "result");

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCH_ITERS; i++) {
            auto ln = ttnn::layer_norm(input, 1e-5f, weight, bias);
            auto tc = ttnn::typecast(ln, DataType::BFLOAT8_B);
        }
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        double dram_ms = std::chrono::duration<double, std::milli>(end - start).count() / BENCH_ITERS;
        fmt::print("  Time: {:.3f} ms\n\n", dram_ms);
    }

    fmt::print("=== Method 2: L1 Interleaved ===\n");
    fmt::print("LayerNorm(L1) → Typecast(L1)\n");
    {
        // LayerNorm outputs to L1
        auto ln_out = ttnn::layer_norm(input, 1e-5f, weight, bias, std::nullopt, L1_MEMORY_CONFIG);
        Finish(cq);
        print_memory_config(ln_out, "ln_out");

        // Typecast reads from L1, outputs to L1
        auto result = ttnn::typecast(ln_out, DataType::BFLOAT8_B, L1_MEMORY_CONFIG);
        Finish(cq);
        print_memory_config(result, "result");

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCH_ITERS; i++) {
            auto ln = ttnn::layer_norm(input, 1e-5f, weight, bias, std::nullopt, L1_MEMORY_CONFIG);
            auto tc = ttnn::typecast(ln, DataType::BFLOAT8_B, L1_MEMORY_CONFIG);
        }
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        double l1_ms = std::chrono::duration<double, std::milli>(end - start).count() / BENCH_ITERS;
        fmt::print("  Time: {:.3f} ms\n\n", l1_ms);
    }

    fmt::print("=== Method 3: L1 Input → L1 Chain ===\n");
    fmt::print("Input(L1) → LayerNorm(L1) → Typecast(L1)\n");
    {
        // Move input to L1 first
        auto input_l1 = ttnn::to_memory_config(input, L1_MEMORY_CONFIG);
        Finish(cq);
        print_memory_config(input_l1, "input_l1");

        // LayerNorm in L1
        auto ln_out = ttnn::layer_norm(input_l1, 1e-5f, weight, bias, std::nullopt, L1_MEMORY_CONFIG);
        Finish(cq);
        print_memory_config(ln_out, "ln_out");

        // Typecast in L1
        auto result = ttnn::typecast(ln_out, DataType::BFLOAT8_B, L1_MEMORY_CONFIG);
        Finish(cq);
        print_memory_config(result, "result");

        // Benchmark (excluding initial to_memory_config)
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCH_ITERS; i++) {
            auto ln = ttnn::layer_norm(input_l1, 1e-5f, weight, bias, std::nullopt, L1_MEMORY_CONFIG);
            auto tc = ttnn::typecast(ln, DataType::BFLOAT8_B, L1_MEMORY_CONFIG);
        }
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        double l1_full_ms = std::chrono::duration<double, std::milli>(end - start).count() / BENCH_ITERS;
        fmt::print("  Time: {:.3f} ms\n\n", l1_full_ms);
    }

    fmt::print("=== COMPLETE ===\n");

    // Cleanup
    Finish(cq);
    ttnn::close_device(*device);

    return 0;
}
