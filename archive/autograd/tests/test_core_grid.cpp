// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Test: Prove and control core grid usage in matmul
// Demonstrates how ttnn::matmul distributes work across cores

#include "common.hpp"
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/types.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#endif

using namespace ttnn;

// Run matmul with specified core grid and measure time
double run_matmul_timed(
    const tt::tt_metal::Tensor& a,
    const tt::tt_metal::Tensor& b,
    test::MeshDevice& device,
    std::optional<CoreGrid> core_grid,
    const std::string& label,
    int warmup = 3,
    int iters = 10
) {
    auto& cq = device.mesh_command_queue();

    // Warmup
    for (int i = 0; i < warmup; i++) {
        auto c = ttnn::matmul(a, b, false, false, std::nullopt, std::nullopt,
                             std::nullopt, std::nullopt, std::nullopt, core_grid);
        Finish(cq);
    }

    // Timed runs
    std::vector<double> times_ms;
    for (int i = 0; i < iters; i++) {
#ifdef TRACY_ENABLE
        ZoneScopedN("matmul_iteration");
#endif
        auto start = std::chrono::high_resolution_clock::now();
        auto c = ttnn::matmul(a, b, false, false, std::nullopt, std::nullopt,
                             std::nullopt, std::nullopt, std::nullopt, core_grid);
        Finish(cq);
        auto end = std::chrono::high_resolution_clock::now();
        times_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    // Calculate mean
    double sum = 0;
    for (auto t : times_ms) sum += t;
    double mean = sum / times_ms.size();

    std::cout << label << ": " << std::fixed << std::setprecision(3) << mean << " ms (avg over " << iters << " runs)\n";
    return mean;
}

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    // Query device core grid
    auto grid = device.compute_with_storage_grid_size();
    std::cout << "Device compute grid: " << grid.x << " x " << grid.y << " = " << (grid.x * grid.y) << " cores\n\n";

    // Create test tensors: 1024x512 @ 512x256 = 1024x256
    const uint32_t M = 1024, K = 512, N = 256;
    std::cout << "Matmul: [" << M << "," << K << "] @ [" << K << "," << N << "] = [" << M << "," << N << "]\n\n";

    auto a = ttnn::ones(ttnn::Shape({M, K}), DataType::BFLOAT16, Layout::TILE, device);
    auto b = ttnn::ones(ttnn::Shape({K, N}), DataType::BFLOAT16, Layout::TILE, device);

    // Run with different core grids
    std::cout << "=== Core Grid Comparison ===\n";

    // Default (auto)
    run_matmul_timed(a, b, device, std::nullopt, "Default (auto)    ");

    // 2x2 = 4 cores
    run_matmul_timed(a, b, device, CoreGrid(2, 2), "CoreGrid(2,2)  4c ");

    // 4x4 = 16 cores
    run_matmul_timed(a, b, device, CoreGrid(4, 4), "CoreGrid(4,4) 16c ");

    // 8x4 = 32 cores
    run_matmul_timed(a, b, device, CoreGrid(8, 4), "CoreGrid(8,4) 32c ");

    // 8x7 = 56 cores (close to max on Wormhole)
    run_matmul_timed(a, b, device, CoreGrid(8, 7), "CoreGrid(8,7) 56c ");

    std::cout << "\n=== Test Complete ===\n";
    std::cout << "Use 'make core-grid-profile' to capture Tracy trace and verify core usage\n";

    return 0;
}
