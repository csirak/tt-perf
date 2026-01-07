// Custom Typecast Kernel - BFP8 → BF16
// Goal: Own this kernel for optimization experiments
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    // Single device (unit mesh)
    auto device = MeshDevice::create_unit_mesh(0);
    MeshCommandQueue& cq = device->mesh_command_queue();

    // Tile sizes
    // BFP8: 1024 data bytes + 64 exponent bytes = 1088 bytes per tile
    // BF16: 32x32 * 2 = 2048 bytes per tile
    constexpr uint32_t TILE_HW = 32 * 32;
    constexpr uint32_t BFP8_TILE_SIZE = 1024 + 64;
    constexpr uint32_t BF16_TILE_SIZE = sizeof(bfloat16) * TILE_HW;

    // DRAM buffers
    DeviceLocalBufferConfig src_cfg{.page_size = BFP8_TILE_SIZE, .buffer_type = BufferType::DRAM};
    DeviceLocalBufferConfig dst_cfg{.page_size = BF16_TILE_SIZE, .buffer_type = BufferType::DRAM};
    ReplicatedBufferConfig buf_cfg{.size = BFP8_TILE_SIZE};  // 1 tile
    ReplicatedBufferConfig dst_buf_cfg{.size = BF16_TILE_SIZE};

    auto src = MeshBuffer::create(buf_cfg, src_cfg, device.get());
    auto dst = MeshBuffer::create(dst_buf_cfg, dst_cfg, device.get());

    // Program on core (0,0)
    Program prog = CreateProgram();
    CoreCoord core = {0, 0};

    // Circular buffers with different formats
    auto cb_in = CircularBufferConfig(BFP8_TILE_SIZE * 2, {{tt::CBIndex::c_0, tt::DataFormat::Bfp8_b}})
        .set_page_size(tt::CBIndex::c_0, BFP8_TILE_SIZE);
    CreateCircularBuffer(prog, core, cb_in);

    auto cb_out = CircularBufferConfig(BF16_TILE_SIZE * 2, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
        .set_page_size(tt::CBIndex::c_16, BF16_TILE_SIZE);
    CreateCircularBuffer(prog, core, cb_out);

    // Kernels (paths relative to eltwise_binary/ since we reuse that build slot)
    auto reader = CreateKernel(prog,
        OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/reader.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto writer = CreateKernel(prog,
        OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/dataflow/writer.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    [[maybe_unused]] auto compute = CreateKernel(prog,
        OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/compute/typecast.cpp", core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

    // Runtime args
    SetRuntimeArgs(prog, reader, core, {src->address(), 1});  // 1 tile
    SetRuntimeArgs(prog, writer, core, {dst->address(), 1});

    // Execute
    MeshWorkload workload;
    workload.add_program(MeshCoordinateRange(device->shape()), std::move(prog));
    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    fmt::print("PASS: Custom typecast kernel executed\n");

    device->close();
    return 0;
}
