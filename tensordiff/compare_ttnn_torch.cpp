// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Simple TTNN ↔ Torch comparison bootstrap (TTNN side).
// Creates a CPU tensor, moves to TTNN, round-trips back to CPU, and saves both.

#include "tensordiff/core/cpu_tensor.hpp"
#include "tensordiff/core/serialize.hpp"
#include "tensordiff/core/compare.hpp"
#include "tensordiff/core/ttnn_bridge.hpp"

#include <ttnn/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>

#include <cstdlib>
#include <filesystem>

using tensordiff::core::CpuTensor;
using tensordiff::core::DType;
using tensordiff::core::compare_tensors;
using tensordiff::core::save_tensor;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;
using DispatchCoreType = tt::tt_metal::DispatchCoreType;

struct DeviceGuard {
    std::shared_ptr<MeshDevice> device;

    DeviceGuard()
        : device([]() {
            bool use_worker = std::getenv("USE_WORKER_DISPATCH") != nullptr;
            return MeshDevice::create_unit_mesh(
                0,
                DEFAULT_L1_SMALL_SIZE,
                0,
                1,
                use_worker ? DispatchCoreConfig{}
                           : DispatchCoreConfig{DispatchCoreType::ETH}
            );
        }()) {
        auto grid = device->compute_with_storage_grid_size();
        fmt::print("# Compute grid: {}x{} = {} cores\n", grid.x, grid.y, grid.x * grid.y);
    }

    ~DeviceGuard() {
        tt::tt_metal::distributed::Finish(device->mesh_command_queue());
        ttnn::close_device(*device);
    }

    MeshDevice& get() { return *device; }
};

int main() {
    DeviceGuard guard;
    auto& device = guard.get();

    const char* out_env = std::getenv("TENSORDIFF_OUT_DIR");
    std::filesystem::path out_dir = out_env ? out_env : "tensordiff/outputs/ttnn";
    std::filesystem::create_directories(out_dir);

    std::vector<uint32_t> shape = {32, 32};
    CpuTensor cpu(shape, DType::kBF16);
    cpu.fill_ramp(-1.0f, 0.01f);

    auto ttnn_tensor = tensordiff::core::to_ttnn(cpu, device, false);
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    auto cpu_roundtrip = tensordiff::core::from_ttnn(ttnn_tensor);

    save_tensor(cpu, (out_dir / "cpu_ref.bin").string(), "cpu");
    save_tensor(cpu_roundtrip, (out_dir / "ttnn_roundtrip.bin").string(), "ttnn");

    auto result = compare_tensors("ttnn_roundtrip", cpu, cpu_roundtrip);
    fmt::print("# tensordiff: {} elements\n", result.numel);
    fmt::print("# max_abs={:.6f} mean_abs={:.6f} rel_l2={:.6f}\n",
               result.max_abs, result.mean_abs, result.rel_l2);
    fmt::print("# wrote: {}\n", out_dir.string());

    return 0;
}
