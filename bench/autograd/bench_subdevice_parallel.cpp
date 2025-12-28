// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: Sub-Device Parallel Dispatch PoC
// Tests whether 2 command queues + 2 sub-devices enable true intra-device parallelism

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <fmt/core.h>
#include <chrono>
#include <thread>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

constexpr uint32_t LOCAL_L1_SIZE = 3200;

// Blank kernel - just exits immediately (minimal dispatch overhead test)
const std::string BLANK_KERNEL = "tt_metal/kernels/dataflow/blank.cpp";

int main() {
    fmt::print("# Sub-Device Parallel Dispatch PoC\n");
    fmt::print("# Testing 2 CQs + 2 sub-devices for intra-device parallelism\n\n");

    // Create device with 2 command queues
    auto device = MeshDevice::create_unit_mesh(
        0,                          // device_id
        LOCAL_L1_SIZE,
        128 * 1024 * 1024,          // trace_region_size
        2,                          // num_command_queues = 2
        DispatchCoreConfig{}
    );

    auto grid_size = device->compute_with_storage_grid_size();
    fmt::print("Device grid: {}x{} = {} cores\n", grid_size.x, grid_size.y, grid_size.x * grid_size.y);
    fmt::print("Command queues: 2\n\n");

    // Split grid into 2 sub-devices (left half and right half)
    uint32_t half_x = grid_size.x / 2;
    CoreRange left_cores({0, 0}, {half_x - 1, grid_size.y - 1});
    CoreRange right_cores({half_x, 0}, {grid_size.x - 1, grid_size.y - 1});

    fmt::print("Sub-device 0: cores ({},{}) to ({},{}) = {} cores\n",
               0, 0, half_x - 1, grid_size.y - 1, half_x * grid_size.y);
    fmt::print("Sub-device 1: cores ({},{}) to ({},{}) = {} cores\n",
               half_x, 0, grid_size.x - 1, grid_size.y - 1, half_x * grid_size.y);

    // Create sub-devices
    SubDevice sub_device_0(std::array{CoreRangeSet(left_cores)});
    SubDevice sub_device_1(std::array{CoreRangeSet(right_cores)});

    // Create and load sub-device manager
    auto sub_device_manager = device->create_sub_device_manager(
        {sub_device_0, sub_device_1},
        LOCAL_L1_SIZE
    );
    device->load_sub_device_manager(sub_device_manager);
    fmt::print("\nLoaded sub-device manager\n");

    // Get worker cores for each sub-device
    auto cores_0 = device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0});
    auto cores_1 = device->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{1});

    fmt::print("Sub-device 0: {} worker cores\n", cores_0.num_cores());
    fmt::print("Sub-device 1: {} worker cores\n", cores_1.num_cores());

    // Create programs for each sub-device
    fmt::print("\nCreating programs...\n");

    Program program_0 = CreateProgram();
    CreateKernel(
        program_0,
        BLANK_KERNEL,
        cores_0,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    fmt::print("Created program_0 with kernel on {} cores\n", cores_0.num_cores());

    Program program_1 = CreateProgram();
    CreateKernel(
        program_1,
        BLANK_KERNEL,
        cores_1,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    fmt::print("Created program_1 with kernel on {} cores\n", cores_1.num_cores());

    // Wrap programs in MeshWorkloads
    MeshWorkload workload_0, workload_1;
    MeshCoordinate zero_coord = MeshCoordinate::zero_coordinate(device->shape().dims());
    MeshCoordinateRange device_range = MeshCoordinateRange(zero_coord, zero_coord);

    workload_0.add_program(device_range, std::move(program_0));
    workload_1.add_program(device_range, std::move(program_1));

    fmt::print("\nWrapped programs in MeshWorkloads\n");

    // Get command queue
    auto& cq = device->mesh_command_queue();

    // Dispatch sequentially and measure time
    constexpr int N_ITERS = 100;
    fmt::print("\nDispatching {} iterations sequentially...\n", N_ITERS);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N_ITERS; i++) {
        EnqueueMeshWorkload(cq, workload_0, false);
        EnqueueMeshWorkload(cq, workload_1, false);
    }
    Finish(cq);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double per_iter_ms = elapsed_ms / N_ITERS;

    double seq_per_iter_ms = per_iter_ms;
    fmt::print("\n## Test 1: Sequential Results (1 CQ)\n");
    fmt::print("Total time: {:.3f} ms for {} iterations\n", elapsed_ms, N_ITERS);
    fmt::print("Per iteration (2 programs): {:.3f} ms\n", per_iter_ms);
    fmt::print("Per program: {:.3f} ms\n", per_iter_ms / 2);

    // ============================================================
    // Test 2: Parallel dispatch with 2 CQs and 2 threads
    // ============================================================
    fmt::print("\n## Test 2: Parallel Dispatch (2 CQs, 2 threads)\n");

    // Get the second command queue
    auto& cq0 = device->mesh_command_queue(0);
    auto& cq1 = device->mesh_command_queue(1);
    fmt::print("Got command queues 0 and 1\n");

    // Dispatch in parallel using 2 threads
    fmt::print("Dispatching {} iterations in parallel...\n", N_ITERS);

    start = std::chrono::high_resolution_clock::now();

    std::thread t0([&]() {
        for (int i = 0; i < N_ITERS; i++) {
            EnqueueMeshWorkload(cq0, workload_0, false);
        }
        Finish(cq0);
    });

    std::thread t1([&]() {
        for (int i = 0; i < N_ITERS; i++) {
            EnqueueMeshWorkload(cq1, workload_1, false);
        }
        Finish(cq1);
    });

    t0.join();
    t1.join();

    end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    per_iter_ms = elapsed_ms / N_ITERS;

    double par_per_iter_ms = per_iter_ms;
    fmt::print("\n## Parallel Results (2 CQs, 2 threads)\n");
    fmt::print("Total time: {:.3f} ms for {} iterations\n", elapsed_ms, N_ITERS);
    fmt::print("Per iteration (2 programs in parallel): {:.3f} ms\n", per_iter_ms);
    fmt::print("Per program (if truly parallel): {:.3f} ms\n", per_iter_ms);

    // ============================================================
    // Test 3: Sequential dispatch with 2 CQs (single thread)
    // This proves if speedup is from parallelism or just from having 2 CQs
    // ============================================================
    fmt::print("\n## Test 3: Sequential Dispatch (2 CQs, 1 thread)\n");
    fmt::print("Dispatching {} iterations sequentially on 2 CQs...\n", N_ITERS);

    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N_ITERS; i++) {
        EnqueueMeshWorkload(cq0, workload_0, false);
        EnqueueMeshWorkload(cq1, workload_1, false);
    }
    Finish(cq0);
    Finish(cq1);

    end = std::chrono::high_resolution_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    per_iter_ms = elapsed_ms / N_ITERS;

    double seq_2cq_per_iter_ms = per_iter_ms;
    fmt::print("\n## Sequential 2-CQ Results (2 CQs, 1 thread)\n");
    fmt::print("Total time: {:.3f} ms for {} iterations\n", elapsed_ms, N_ITERS);
    fmt::print("Per iteration (2 programs): {:.3f} ms\n", per_iter_ms);

    // ============================================================
    // Summary
    // ============================================================
    fmt::print("\n## Summary\n");
    fmt::print("| Config                       | ms/iter | vs 1-CQ seq |\n");
    fmt::print("|------------------------------|---------|-------------|\n");
    fmt::print("| Sequential (1 CQ, 1 thread)  | {:.3f}   | 1.00x       |\n", seq_per_iter_ms);
    fmt::print("| Sequential (2 CQs, 1 thread) | {:.3f}   | {:.2f}x       |\n", seq_2cq_per_iter_ms, seq_per_iter_ms / seq_2cq_per_iter_ms);
    fmt::print("| Parallel (2 CQs, 2 threads)  | {:.3f}   | {:.2f}x       |\n", par_per_iter_ms, seq_per_iter_ms / par_per_iter_ms);

    fmt::print("\n## Analysis\n");
    double speedup_2cq = seq_per_iter_ms / seq_2cq_per_iter_ms;
    double speedup_threads = seq_2cq_per_iter_ms / par_per_iter_ms;

    if (speedup_threads > 1.3) {
        fmt::print("Parallelism speedup (threads): {:.2f}x\n", speedup_threads);
        fmt::print("2-CQ overhead reduction: {:.2f}x\n", speedup_2cq);
        fmt::print("=> Speedup is from PARALLELISM (threads enable concurrent dispatch)\n");
    } else {
        fmt::print("2-CQ speedup: {:.2f}x\n", speedup_2cq);
        fmt::print("Thread parallelism: {:.2f}x (negligible)\n", speedup_threads);
        fmt::print("=> Speedup is from 2 CQs, NOT from parallelism\n");
    }

    device->close();
    return 0;
}
