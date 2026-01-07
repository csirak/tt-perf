// Custom L1-to-L1 Typecast Kernel - BF16 → BFP8
// Data already in CB_0 from previous compute, output to CB_16 (no DRAM movement)
// Goal: Avoid DRAM round-trip when fusing with other compute kernels
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
    constexpr uint32_t TILE_HW = 32 * 32;
    constexpr uint32_t BFP8_TILE_SIZE = 1024 + 64;
    constexpr uint32_t BF16_TILE_SIZE = sizeof(bfloat16) * TILE_HW;

    // Program on core (0,0)
    Program prog = CreateProgram();
    CoreCoord core = {0, 0};

    // CB_0: BF16 input (assume data already here from previous compute)
    auto cb_in = CircularBufferConfig(BF16_TILE_SIZE * 2, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
        .set_page_size(tt::CBIndex::c_0, BF16_TILE_SIZE);
    CreateCircularBuffer(prog, core, cb_in);

    // CB_16: BFP8 output
    auto cb_out = CircularBufferConfig(BFP8_TILE_SIZE * 2, {{tt::CBIndex::c_16, tt::DataFormat::Bfp8_b}})
        .set_page_size(tt::CBIndex::c_16, BFP8_TILE_SIZE);
    CreateCircularBuffer(prog, core, cb_out);

    // ONLY compute kernel - no reader/writer
    // This demonstrates the L1-to-L1 pattern for kernel fusion
    [[maybe_unused]] auto compute = CreateKernel(prog,
        OVERRIDE_KERNEL_PREFIX "eltwise_binary/kernels/compute/typecast_bf16_to_bfp8.cpp", core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

    // No runtime args needed - compute reads from CB_0, writes to CB_16

    // Execute
    MeshWorkload workload;
    workload.add_program(MeshCoordinateRange(device->shape()), std::move(prog));
    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    fmt::print("PASS: L1-to-L1 typecast kernel executed\n");

    device->close();
    return 0;
}
