// Typecast Benchmark - Measure BFP8 <-> BF16 conversion overhead
// Goal: Understand why typecast consumes 41% of backward pass time

#include <chrono>
#include <vector>

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <tt-metalium/distributed.hpp>

#include <fmt/core.h>

using namespace ttnn;
using namespace tt::tt_metal::distributed;

// Wormhole N300 DRAM bandwidth: 288 GB/s
constexpr double DRAM_BW_GBS = 288.0;

struct BenchResult {
    std::string shape_str;
    std::string direction;
    double time_ms;
    double throughput_gbs;
    double dram_util_pct;
    size_t num_elements;
};

double get_tensor_bytes(size_t elements, DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16: return elements * 2.0;
        case DataType::BFLOAT8_B: return elements * 1.0;
        case DataType::BFLOAT4_B: return elements * 0.5;
        default: return elements * 2.0;
    }
}

std::string shape_to_string(const ttnn::Shape& shape) {
    std::string s = "[";
    for (size_t i = 0; i < shape.rank(); i++) {
        if (i > 0) s += ",";
        s += std::to_string(shape[i]);
    }
    s += "]";
    return s;
}

BenchResult benchmark_typecast(
    MeshDevice& device,
    const ttnn::Shape& shape,
    DataType src_dtype,
    DataType dst_dtype,
    int warmup_iters = 3,
    int bench_iters = 10
) {
    MeshCommandQueue& cq = device.mesh_command_queue();

    // Create source tensor on device
    auto src = ones(shape, src_dtype, TILE_LAYOUT, device);
    size_t num_elements = src.logical_volume();
    Finish(cq);

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        auto dst = ttnn::typecast(src, dst_dtype);
    }
    Finish(cq);

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < bench_iters; i++) {
        auto dst = ttnn::typecast(src, dst_dtype);
    }
    Finish(cq);
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / bench_iters;

    // Calculate throughput
    // For typecast: read src + write dst
    double src_bytes = get_tensor_bytes(num_elements, src_dtype);
    double dst_bytes = get_tensor_bytes(num_elements, dst_dtype);
    double total_bytes = src_bytes + dst_bytes;
    double throughput_gbs = (total_bytes / 1e9) / (avg_ms / 1000.0);
    double dram_util = (throughput_gbs / DRAM_BW_GBS) * 100.0;

    std::string direction;
    if (src_dtype == DataType::BFLOAT8_B && dst_dtype == DataType::BFLOAT16) {
        direction = "BFP8->BF16";
    } else if (src_dtype == DataType::BFLOAT16 && dst_dtype == DataType::BFLOAT8_B) {
        direction = "BF16->BFP8";
    } else {
        direction = "other";
    }

    return BenchResult{
        shape_to_string(shape),
        direction,
        avg_ms,
        throughput_gbs,
        dram_util,
        num_elements
    };
}

int main() {
    fmt::print("=== TYPECAST BENCHMARK ===\n");
    fmt::print("Goal: Understand BFP8 <-> BF16 conversion overhead\n");
    fmt::print("DRAM BW: {} GB/s\n\n", DRAM_BW_GBS);

    // Open device using same pattern as other benchmarks
    using tt::tt_metal::DispatchCoreConfig;
    using tt::tt_metal::DispatchCoreType;
    auto device = MeshDevice::create_unit_mesh(
        0,                           // device_id
        DEFAULT_L1_SMALL_SIZE,       // l1_small_size
        128 * 1024 * 1024,           // trace_region_size
        1,                           // num_command_queues
        DispatchCoreConfig{DispatchCoreType::ETH}
    );

    // Print grid info
    auto grid = device->compute_with_storage_grid_size();
    fmt::print("Compute grid: {}x{} = {} cores\n\n", grid.x, grid.y, grid.x * grid.y);

    // GPT-2 relevant shapes
    std::vector<ttnn::Shape> shapes = {
        ttnn::Shape({1, 1, 1024, 768}),      // GPT-2: hidden activations (786K)
        ttnn::Shape({1, 12, 1024, 64}),      // GPT-2: attention per head (786K)
        ttnn::Shape({1, 1, 1024, 3072}),     // GPT-2: MLP intermediate (3.1M)
        ttnn::Shape({32, 1, 1024, 768}),     // GPT-2: batch=32 hidden (25M)
    };

    std::vector<BenchResult> results;

    // BFP8 -> BF16 (the expensive direction in backward)
    fmt::print("Testing BFP8 -> BF16 (backward direction)...\n");
    for (const auto& shape : shapes) {
        auto result = benchmark_typecast(*device, shape, DataType::BFLOAT8_B, DataType::BFLOAT16);
        results.push_back(result);
        fmt::print("  {}: {:.3f} ms, {:.1f} GB/s ({:.1f}% DRAM)\n",
                   result.shape_str, result.time_ms, result.throughput_gbs, result.dram_util_pct);
    }

    fmt::print("\nTesting BF16 -> BFP8 (forward direction)...\n");
    for (const auto& shape : shapes) {
        auto result = benchmark_typecast(*device, shape, DataType::BFLOAT16, DataType::BFLOAT8_B);
        results.push_back(result);
        fmt::print("  {}: {:.3f} ms, {:.1f} GB/s ({:.1f}% DRAM)\n",
                   result.shape_str, result.time_ms, result.throughput_gbs, result.dram_util_pct);
    }

    // Print summary table
    fmt::print("\n=== SUMMARY ===\n");
    fmt::print("{:<22} {:<14} {:>10} {:>12} {:>12} {:>14}\n",
               "Shape", "Direction", "Time(ms)", "GB/s", "DRAM%", "Elements");
    fmt::print("{}\n", std::string(84, '-'));

    for (const auto& r : results) {
        fmt::print("{:<22} {:<14} {:>10.3f} {:>12.1f} {:>12.1f} {:>14}\n",
                   r.shape_str, r.direction, r.time_ms, r.throughput_gbs, r.dram_util_pct, r.num_elements);
    }

    // CSV output
    fmt::print("\n=== CSV ===\n");
    fmt::print("shape,direction,time_ms,throughput_gbs,dram_util_pct,elements\n");
    for (const auto& r : results) {
        fmt::print("{},{},{:.4f},{:.2f},{:.2f},{}\n",
                   r.shape_str, r.direction, r.time_ms, r.throughput_gbs, r.dram_util_pct, r.num_elements);
    }

    // Cleanup
    Finish(device->mesh_command_queue());
    ttnn::close_device(*device);

    fmt::print("\nDone.\n");
    return 0;
}
