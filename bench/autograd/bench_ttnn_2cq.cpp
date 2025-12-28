// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark: Real TTNN Ops with 2 Command Queues
// Tests whether dispatching matmul to different CQs provides speedup
//
// KEY FINDING: Without sub-devices, there's only 1 sub-device owned by CQ 0.
// CQ 1 cannot use it without ownership transfer. For true parallel dispatch,
// we need 2 sub-devices (split the grid) with each CQ owning its own.

#include <ttnn/device.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/core.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/sub_device.hpp>
#include <fmt/core.h>
#include <chrono>
#include <thread>

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

constexpr int N_WARMUP = 5;
constexpr int N_ITERS = 50;

int main() {
    fmt::print("# TTNN 2-CQ Benchmark: Real Matmul Operations\n");
    fmt::print("# Testing if queue_id dispatch provides speedup for real ops\n\n");

    // Create device with 2 command queues
    auto device = MeshDevice::create_unit_mesh(
        0,                          // device_id
        DEFAULT_L1_SMALL_SIZE,
        128 * 1024 * 1024,          // trace_region_size
        2,                          // num_command_queues = 2
        DispatchCoreConfig{}
    );

    auto grid_size = device->compute_with_storage_grid_size();
    fmt::print("Device grid: {}x{} = {} cores\n", grid_size.x, grid_size.y, grid_size.x * grid_size.y);
    fmt::print("Command queues: 2\n");
    fmt::print("Warmup: {}, Iterations: {}\n\n", N_WARMUP, N_ITERS);

    // ============================================================
    // Test 1: Sequential on 1 CQ (baseline) - Full grid
    // ============================================================
    fmt::print("## Test 1: Sequential (1 CQ, full grid)\n");

    constexpr uint32_t M = 512, K = 512, N = 512;
    fmt::print("Matmul shape: [{}x{}] @ [{}x{}]\n", M, K, K, N);

    auto a = ttnn::full(ttnn::Shape({M, K}), 1.0f, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device);
    auto b = ttnn::full(ttnn::Shape({K, N}), 2.0f, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device);

    // Warmup
    for (int i = 0; i < N_WARMUP; i++) {
        auto c = ttnn::matmul(a, b);
    }
    Synchronize(device.get(), std::nullopt);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_ITERS; i++) {
        auto c0 = ttnn::matmul(a, b);
        auto c1 = ttnn::matmul(a, b);
    }
    Synchronize(device.get(), std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    double seq_1cq_ms = std::chrono::duration<double, std::milli>(end - start).count() / N_ITERS;
    fmt::print("Per iteration (2 matmuls): {:.3f} ms\n", seq_1cq_ms);

    // ============================================================
    // NOTE: Tests 2 and 3 require sub-device setup for 2 CQ dispatch
    // Without sub-devices, CQ 1 cannot use sub-device 0 (owned by CQ 0)
    // ============================================================
    fmt::print("\n## Note on 2-CQ TTNN dispatch\n");
    fmt::print("Without sub-devices, there's only 1 sub-device owned by CQ 0.\n");
    fmt::print("CQ 1 cannot dispatch without ownership transfer via Finish/events.\n");
    fmt::print("For true parallel TTNN dispatch, need:\n");
    fmt::print("  1. Create 2 sub-devices (split grid)\n");
    fmt::print("  2. Allocate tensors on each sub-device\n");
    fmt::print("  3. Each CQ owns its own sub-device\n");
    fmt::print("\nThis is fundamentally different from the blank kernel test\n");
    fmt::print("which used TT-Metal Programs directly (not TTNN ops).\n");

    // ============================================================
    // Test 2: Same workload but with sub-device setup for comparison
    // ============================================================
    fmt::print("\n## Test 2: Setting up 2 sub-devices for parallel dispatch...\n");

    // Split grid into 2 sub-devices (left half and right half)
    uint32_t half_x = grid_size.x / 2;
    CoreRange left_cores({0, 0}, {half_x - 1, grid_size.y - 1});
    CoreRange right_cores({half_x, 0}, {grid_size.x - 1, grid_size.y - 1});

    fmt::print("Sub-device 0: cores ({},{}) to ({},{}) = {} cores\n",
               0, 0, half_x - 1, grid_size.y - 1, half_x * grid_size.y);
    fmt::print("Sub-device 1: cores ({},{}) to ({},{}) = {} cores\n",
               half_x, 0, grid_size.x - 1, grid_size.y - 1, (grid_size.x - half_x) * grid_size.y);

    // Create sub-devices
    SubDevice sub_device_0(std::array{CoreRangeSet(left_cores)});
    SubDevice sub_device_1(std::array{CoreRangeSet(right_cores)});

    auto sub_device_manager = device->create_sub_device_manager(
        {sub_device_0, sub_device_1},
        DEFAULT_L1_SMALL_SIZE
    );
    device->load_sub_device_manager(sub_device_manager);
    fmt::print("Loaded sub-device manager with 2 sub-devices\n");

    // TODO: TTNN doesn't currently support sub_device_id in MemoryConfig
    // This means we can't allocate tensors on specific sub-devices via TTNN API
    // Would need tt-metal level tensor allocation with sub-device specification

    fmt::print("\n## Limitation: TTNN API doesn't expose sub_device_id in MemoryConfig\n");
    fmt::print("Cannot allocate TTNN tensors on specific sub-devices yet.\n");
    fmt::print("True parallel TTNN dispatch requires TTNN API changes.\n");

    // ============================================================
    // Summary
    // ============================================================
    fmt::print("\n## Summary\n");
    fmt::print("| Config                  | ms/iter | Notes                           |\n");
    fmt::print("|-------------------------|---------|--------------------------------|\n");
    fmt::print("| Sequential (1 CQ)       | {:.3f}   | Full grid, works fine          |\n", seq_1cq_ms);
    fmt::print("| 2 CQs without subdevice | N/A     | CQ ownership conflict          |\n");
    fmt::print("| 2 CQs with subdevices   | N/A     | TTNN API doesn't support yet   |\n");

    fmt::print("\n## Conclusion\n");
    fmt::print("The 4.8x speedup from 2 CQs was measured on BLANK KERNELS\n");
    fmt::print("using TT-Metal Program API, not real TTNN operations.\n");
    fmt::print("For TTNN ops, 2 CQ parallel dispatch requires:\n");
    fmt::print("  - Sub-device grid partitioning\n");
    fmt::print("  - TTNN MemoryConfig with sub_device_id support\n");
    fmt::print("  - Tensor allocation on specific sub-devices\n");
    fmt::print("These features are not yet exposed in the TTNN high-level API.\n");

    device->close();
    return 0;
}
