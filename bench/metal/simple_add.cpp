// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Minimal element-wise add: src0 + src1 -> dst (single tile, single core)

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/distributed.hpp>
#include <random>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    // Single device (unit mesh)
    auto device = MeshDevice::create_unit_mesh(0);
    MeshCommandQueue& cq = device->mesh_command_queue();

    // Tile config: 32x32 bfloat16 = 2KB
    constexpr uint32_t TILE_HW = 32 * 32;
    constexpr uint32_t TILE_SIZE = sizeof(bfloat16) * TILE_HW;

    // DRAM buffers for src0, src1, dst
    DeviceLocalBufferConfig dram_cfg{.page_size = TILE_SIZE, .buffer_type = BufferType::DRAM};
    ReplicatedBufferConfig buf_cfg{.size = TILE_SIZE};
    auto src0 = MeshBuffer::create(buf_cfg, dram_cfg, device.get());
    auto src1 = MeshBuffer::create(buf_cfg, dram_cfg, device.get());
    auto dst  = MeshBuffer::create(buf_cfg, dram_cfg, device.get());

    // Random input data
    std::vector<bfloat16> a(TILE_HW), b(TILE_HW);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < TILE_HW; i++) {
        a[i] = bfloat16(dist(rng));
        b[i] = bfloat16(dist(rng));
    }
    EnqueueWriteMeshBuffer(cq, src0, a, false);
    EnqueueWriteMeshBuffer(cq, src1, b, false);

    // Program on core (0,0)
    Program prog = CreateProgram();
    CoreCoord core = {0, 0};

    // Circular buffers: c_0 (in0), c_1 (in1), c_16 (out)
    auto make_cb = [&](tt::CBIndex idx) {
        return CircularBufferConfig(TILE_SIZE, {{idx, tt::DataFormat::Float16_b}})
            .set_page_size(idx, TILE_SIZE);
    };
    CreateCircularBuffer(prog, core, make_cb(tt::CBIndex::c_0));
    CreateCircularBuffer(prog, core, make_cb(tt::CBIndex::c_1));
    CreateCircularBuffer(prog, core, make_cb(tt::CBIndex::c_16));

    // Kernels
    auto reader = CreateKernel(prog,
        OVERRIDE_KERNEL_PREFIX "simple_add/kernels/dataflow/reader.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto writer = CreateKernel(prog,
        OVERRIDE_KERNEL_PREFIX "simple_add/kernels/dataflow/writer.cpp", core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    [[maybe_unused]] auto compute = CreateKernel(prog,
        OVERRIDE_KERNEL_PREFIX "simple_add/kernels/compute/add.cpp", core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

    // Runtime args: buffer addresses
    SetRuntimeArgs(prog, reader, core, {src0->address(), src1->address()});
    SetRuntimeArgs(prog, writer, core, {dst->address()});

    // Execute
    MeshWorkload workload;
    workload.add_program(MeshCoordinateRange(device->shape()), std::move(prog));
    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);

    // Validate
    std::vector<bfloat16> result;
    EnqueueReadMeshBuffer(cq, result, dst, true);

    bool pass = true;
    for (size_t i = 0; i < TILE_HW; i++) {
        float expected = float(a[i]) + float(b[i]);
        if (std::abs(expected - float(result[i])) > 0.01f) {
            fmt::print(stderr, "FAIL at {}: {} + {} = {} (got {})\n",
                i, float(a[i]), float(b[i]), expected, float(result[i]));
            pass = false;
            break;
        }
    }

    device->close();
    if (pass) {
        fmt::print("PASS\n");
    } else {
        fmt::print("FAIL\n");
    }
    return pass ? 0 : 1;
}
