// Benchmark: Custom Metal Kernel vs ttnn::typecast
// Compare BFP8 → BF16 conversion overhead for GPT-2 shape [1,1,1024,768]
#include <chrono>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

constexpr double DRAM_BW_GBS = 288.0;

int main() {
    fmt::print("=== CUSTOM KERNEL BENCHMARK ===\n");
    fmt::print("Shape: [1,1,1024,768] (786432 elements, 768 tiles)\n");
    fmt::print("Compare: Custom Metal kernel vs ttnn::typecast (~0.058 ms)\n\n");

    auto device = MeshDevice::create_unit_mesh(0);
    MeshCommandQueue& cq = device->mesh_command_queue();

    // Tile sizes
    constexpr uint32_t TILE_HW = 32 * 32;
    constexpr uint32_t BFP8_TILE_SIZE = 1024 + 64;  // 1088 bytes
    constexpr uint32_t BF16_TILE_SIZE = sizeof(bfloat16) * TILE_HW;  // 2048 bytes

    // GPT-2 hidden: [1,1,1024,768] = 768 tiles
    constexpr uint32_t NUM_TILES = 768;
    constexpr uint32_t NUM_ELEMENTS = 786432;

    constexpr int BENCH_ITERS = 20;

    // Create buffers
    uint32_t src_size = NUM_TILES * BFP8_TILE_SIZE;
    uint32_t dst_size = NUM_TILES * BF16_TILE_SIZE;

    DeviceLocalBufferConfig src_cfg{.page_size = BFP8_TILE_SIZE, .buffer_type = BufferType::DRAM};
    DeviceLocalBufferConfig dst_cfg{.page_size = BF16_TILE_SIZE, .buffer_type = BufferType::DRAM};
    ReplicatedBufferConfig src_buf_cfg{.size = src_size};
    ReplicatedBufferConfig dst_buf_cfg{.size = dst_size};

    auto src = MeshBuffer::create(src_buf_cfg, src_cfg, device.get());
    auto dst = MeshBuffer::create(dst_buf_cfg, dst_cfg, device.get());

    // Create program
    Program prog = CreateProgram();
    CoreCoord core = {0, 0};

    // Circular buffers - double buffered
    auto cb_in = CircularBufferConfig(BFP8_TILE_SIZE * 2, {{tt::CBIndex::c_0, tt::DataFormat::Bfp8_b}})
        .set_page_size(tt::CBIndex::c_0, BFP8_TILE_SIZE);
    CreateCircularBuffer(prog, core, cb_in);

    auto cb_out = CircularBufferConfig(BF16_TILE_SIZE * 2, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
        .set_page_size(tt::CBIndex::c_16, BF16_TILE_SIZE);
    CreateCircularBuffer(prog, core, cb_out);

    // Kernels - compute kernel has NUM_TILES=768 hardcoded via define
    auto reader = CreateKernel(prog,
        OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/reader.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto writer = CreateKernel(prog,
        OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/writer.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Pass NUM_TILES as compile-time define
    std::map<std::string, std::string> compute_defines = {{"NUM_TILES", std::to_string(NUM_TILES)}};
    [[maybe_unused]] auto compute = CreateKernel(prog,
        OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/compute/typecast.cpp", core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .defines = compute_defines});

    // Runtime args
    SetRuntimeArgs(prog, reader, core, {src->address(), NUM_TILES});
    SetRuntimeArgs(prog, writer, core, {dst->address(), NUM_TILES});

    // Execute once to compile kernels
    fmt::print("Compiling kernels...\n");
    {
        MeshWorkload workload;
        workload.add_program(MeshCoordinateRange(device->shape()), std::move(prog));
        EnqueueMeshWorkload(cq, workload, false);
        Finish(cq);
    }

    // For repeated execution, we need to recreate programs each time
    // or use trace API. Let's just measure single execution time accurately.

    // Run multiple fresh programs for timing
    fmt::print("Benchmarking ({} iterations)...\n", BENCH_ITERS);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < BENCH_ITERS; i++) {
        // Create fresh program each iteration
        Program new_prog = CreateProgram();

        auto new_cb_in = CircularBufferConfig(BFP8_TILE_SIZE * 2, {{tt::CBIndex::c_0, tt::DataFormat::Bfp8_b}})
            .set_page_size(tt::CBIndex::c_0, BFP8_TILE_SIZE);
        CreateCircularBuffer(new_prog, core, new_cb_in);

        auto new_cb_out = CircularBufferConfig(BF16_TILE_SIZE * 2, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_16, BF16_TILE_SIZE);
        CreateCircularBuffer(new_prog, core, new_cb_out);

        auto new_reader = CreateKernel(new_prog,
            OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/reader.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        auto new_writer = CreateKernel(new_prog,
            OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/writer.cpp", core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        std::map<std::string, std::string> new_defines = {{"NUM_TILES", std::to_string(NUM_TILES)}};
        [[maybe_unused]] auto new_compute = CreateKernel(new_prog,
            OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/compute/typecast.cpp", core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .defines = new_defines});

        SetRuntimeArgs(new_prog, new_reader, core, {src->address(), NUM_TILES});
        SetRuntimeArgs(new_prog, new_writer, core, {dst->address(), NUM_TILES});

        MeshWorkload workload;
        workload.add_program(MeshCoordinateRange(device->shape()), std::move(new_prog));
        EnqueueMeshWorkload(cq, workload, false);
    }
    Finish(cq);
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_ms = total_ms / BENCH_ITERS;

    // Throughput: read BFP8 + write BF16
    double bytes_moved = (double)NUM_ELEMENTS * 1.0 + (double)NUM_ELEMENTS * 2.0;  // BFP8=1B, BF16=2B
    double throughput_gbs = (bytes_moved / 1e9) / (avg_ms / 1000.0);
    double dram_util = (throughput_gbs / DRAM_BW_GBS) * 100.0;

    fmt::print("\n=== RESULTS ===\n");
    fmt::print("Custom kernel:    {:.3f} ms\n", avg_ms);
    fmt::print("ttnn::typecast:   0.058 ms (from previous benchmark)\n");
    fmt::print("Throughput:       {:.1f} GB/s ({:.1f}% DRAM utilization)\n", throughput_gbs, dram_util);
    fmt::print("Speedup:          {:.2f}x\n", 0.058 / avg_ms);

    fmt::print("\nDone.\n");
    device->close();
    return 0;
}
