// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Load A/B from TDF1, run TTNN matmul, save roundtrip + output.

#include "tensordiff/core/serialize.hpp"
#include "tensordiff/core/ttnn_bridge.hpp"
#include "tensordiff/core/compare.hpp"

#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>

#include <cstdlib>
#include <filesystem>

using tensordiff::core::CpuTensor;
using tensordiff::core::load_tensor;
using tensordiff::core::save_tensor;
using tensordiff::core::compare_tensors;
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
    std::filesystem::path out_dir = out_env ? out_env : "tensordiff/outputs/matmul_torch_rand";

    auto a = load_tensor((out_dir / "torch_a.bin").string());
    auto b = load_tensor((out_dir / "torch_b.bin").string());

    auto tt_a = tensordiff::core::to_ttnn(a, device, false);
    auto tt_b = tensordiff::core::to_ttnn(b, device, false);
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    auto a_round = tensordiff::core::from_ttnn(tt_a);
    auto b_round = tensordiff::core::from_ttnn(tt_b);

    auto tt_out = ttnn::matmul(tt_a, tt_b);
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    auto out_cpu = tensordiff::core::from_ttnn(tt_out);

    save_tensor(a_round, (out_dir / "ttnn_a_roundtrip.bin").string(), "ttnn");
    save_tensor(b_round, (out_dir / "ttnn_b_roundtrip.bin").string(), "ttnn");
    save_tensor(out_cpu, (out_dir / "ttnn_out.bin").string(), "ttnn");

    auto a_cmp = compare_tensors("a_roundtrip", a, a_round);
    auto b_cmp = compare_tensors("b_roundtrip", b, b_round);
    fmt::print("# a_roundtrip: max_abs={:.6f} rel_l2={:.6f}\n", a_cmp.max_abs, a_cmp.rel_l2);
    fmt::print("# b_roundtrip: max_abs={:.6f} rel_l2={:.6f}\n", b_cmp.max_abs, b_cmp.rel_l2);
    fmt::print("# wrote: {}\n", out_dir.string());

    return 0;
}
