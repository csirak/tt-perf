// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: Sub-Device Parallel Dispatch - Configuration Sweep
// Tests multiple sub-device configurations (1-8) with 2 CQs
//
// Key constraints:
// - Max 8 sub-devices (DISPATCH_MESSAGE_ENTRIES = 8)
// - Max 2 hardware CQs (MAX_NUM_HW_CQS = 2)
// - N sub-devices share 2 CQs via software multiplexing

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <fmt/core.h>
#include <chrono>
#include <thread>
#include <vector>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

constexpr uint32_t LOCAL_L1_SIZE = 3200;
constexpr int N_WARMUP = 5;
constexpr int N_ITERS = 50;

// Blank kernel for minimal dispatch overhead testing
const std::string BLANK_KERNEL = "tt_metal/kernels/dataflow/blank.cpp";

// Create N sub-devices by splitting the grid
std::vector<SubDevice> create_subdevices(uint32_t n_partitions, CoreCoord grid_size) {
    std::vector<SubDevice> sub_devices;
    uint32_t gx = grid_size.x, gy = grid_size.y;

    if (n_partitions == 1) {
        // Full grid
        sub_devices.emplace_back(std::array{CoreRangeSet(CoreRange({0, 0}, {gx - 1, gy - 1}))});
    } else if (n_partitions == 2) {
        // Left/right halves (4×7 each on 8×7 grid)
        uint32_t half = gx / 2;
        sub_devices.emplace_back(std::array{CoreRangeSet(CoreRange({0, 0}, {half - 1, gy - 1}))});
        sub_devices.emplace_back(std::array{CoreRangeSet(CoreRange({half, 0}, {gx - 1, gy - 1}))});
    } else if (n_partitions == 4) {
        // 4 vertical strips (2×7 each on 8×7 grid)
        uint32_t strip_width = gx / 4;
        for (uint32_t i = 0; i < 4; i++) {
            uint32_t sx = i * strip_width;
            sub_devices.emplace_back(std::array{CoreRangeSet(
                CoreRange({sx, 0}, {sx + strip_width - 1, gy - 1}))});
        }
    } else if (n_partitions == 7) {
        // 7 rows (8×1 each on 8×7 grid)
        for (uint32_t row = 0; row < 7; row++) {
            sub_devices.emplace_back(std::array{CoreRangeSet(
                CoreRange({0, row}, {gx - 1, row}))});
        }
    } else if (n_partitions == 8) {
        // 8 columns (1×7 each on 8×7 grid)
        for (uint32_t col = 0; col < 8; col++) {
            sub_devices.emplace_back(std::array{CoreRangeSet(
                CoreRange({col, 0}, {col, gy - 1}))});
        }
    }

    return sub_devices;
}

// Create a program with blank kernel for a sub-device
Program create_blank_program(MeshDevice* device, SubDeviceId sub_device_id) {
    auto cores = device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
    Program program = CreateProgram();
    CreateKernel(
        program,
        BLANK_KERNEL,
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    return program;
}

// Run benchmark: sequential dispatch (1 CQ)
double run_sequential(MeshDevice* device, std::vector<MeshWorkload>& workloads, int n_iters) {
    auto& cq = device->mesh_command_queue();

    // Warmup
    for (int w = 0; w < N_WARMUP; w++) {
        for (auto& workload : workloads) {
            EnqueueMeshWorkload(cq, workload, false);
        }
        Finish(cq);
    }

    // Timed
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_iters; i++) {
        for (auto& workload : workloads) {
            EnqueueMeshWorkload(cq, workload, false);
        }
    }
    Finish(cq);
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Run benchmark: parallel dispatch (2 CQs, 2 threads)
// For n_sd=2: Thread 0 dispatches workload[0], Thread 1 dispatches workload[1]
// For n_sd>2: Split workloads evenly between threads
double run_parallel(MeshDevice* device, std::vector<MeshWorkload>& workloads, int n_iters) {
    auto& cq0 = device->mesh_command_queue(0);
    auto& cq1 = device->mesh_command_queue(1);

    // Warmup - do parallel warmup to set correct CQ ownership
    std::thread w0([&]() {
        for (size_t i = 0; i < workloads.size(); i += 2) {
            EnqueueMeshWorkload(cq0, workloads[i], false);
        }
        Finish(cq0);
    });
    std::thread w1([&]() {
        for (size_t i = 1; i < workloads.size(); i += 2) {
            EnqueueMeshWorkload(cq1, workloads[i], false);
        }
        Finish(cq1);
    });
    w0.join();
    w1.join();

    // Timed - parallel dispatch
    auto start = std::chrono::high_resolution_clock::now();

    std::thread t0([&]() {
        for (int iter = 0; iter < n_iters; iter++) {
            for (size_t i = 0; i < workloads.size(); i += 2) {
                EnqueueMeshWorkload(cq0, workloads[i], false);
            }
        }
        Finish(cq0);
    });

    std::thread t1([&]() {
        for (int iter = 0; iter < n_iters; iter++) {
            for (size_t i = 1; i < workloads.size(); i += 2) {
                EnqueueMeshWorkload(cq1, workloads[i], false);
            }
        }
        Finish(cq1);
    });

    t0.join();
    t1.join();

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Run a single configuration test
void run_config(uint32_t n_sd, double& baseline_time) {
    // Create device with 2 command queues
    auto device = MeshDevice::create_unit_mesh(
        0,                          // device_id
        LOCAL_L1_SIZE,
        128 * 1024 * 1024,          // trace_region_size
        2,                          // num_command_queues = 2
        DispatchCoreConfig{}
    );

    auto grid_size = device->compute_with_storage_grid_size();

    // Create sub-devices
    auto sub_devices = create_subdevices(n_sd, grid_size);
    auto sub_device_manager = device->create_sub_device_manager(sub_devices, LOCAL_L1_SIZE);
    device->load_sub_device_manager(sub_device_manager);

    // Calculate cores per partition
    uint32_t total_cores = grid_size.x * grid_size.y;
    uint32_t cores_per_partition = total_cores / n_sd;

    // Create programs and workloads for each sub-device
    std::vector<MeshWorkload> workloads;
    MeshCoordinate zero_coord = MeshCoordinate::zero_coordinate(device->shape().dims());
    MeshCoordinateRange device_range = MeshCoordinateRange(zero_coord, zero_coord);

    for (uint32_t i = 0; i < n_sd; i++) {
        Program program = create_blank_program(device.get(), SubDeviceId{static_cast<uint8_t>(i)});
        MeshWorkload workload;
        workload.add_program(device_range, std::move(program));
        workloads.push_back(std::move(workload));
    }

    // Test 1: Sequential dispatch
    double seq_time = run_sequential(device.get(), workloads, N_ITERS);
    double seq_per_iter = seq_time / N_ITERS;

    if (n_sd == 1) {
        baseline_time = seq_time;
    }
    double seq_speedup = (baseline_time > 0) ? baseline_time / seq_time : 1.0;

    fmt::print("{},{},sequential,{:.3f},{:.3f},{:.2f}\n",
               n_sd, cores_per_partition, seq_time, seq_per_iter, seq_speedup);

    // Test 2: Parallel dispatch (skip for n_sd == 1)
    if (n_sd > 1) {
        double par_time = run_parallel(device.get(), workloads, N_ITERS);
        double par_per_iter = par_time / N_ITERS;
        double par_speedup = (baseline_time > 0) ? baseline_time / par_time : 1.0;

        fmt::print("{},{},parallel,{:.3f},{:.3f},{:.2f}\n",
                   n_sd, cores_per_partition, par_time, par_per_iter, par_speedup);
    }

    device->close();
}

int main() {
    fmt::print("# Sub-Device Parallel Benchmark Sweep\n");
    fmt::print("# Testing 1-8 sub-devices with 2 CQs\n");
    fmt::print("# Warmup: {}, Timed iters: {}\n", N_WARMUP, N_ITERS);
    fmt::print("#\n");

    // CSV header
    fmt::print("n_subdevices,cores_per_partition,dispatch_mode,total_ms,per_iter_ms,speedup_vs_baseline\n");

    // Configuration sweep - each config gets fresh device
    std::vector<uint32_t> n_subdevices_opts = {1, 2, 4, 7, 8};
    double baseline_time = 0.0;

    for (auto n_sd : n_subdevices_opts) {
        run_config(n_sd, baseline_time);
    }

    fmt::print("#\n");
    fmt::print("# Done!\n");

    return 0;
}
