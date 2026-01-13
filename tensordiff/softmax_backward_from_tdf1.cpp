// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Softmax backward binary-op test: d_scores = attn * (d_attn - sum(d_attn * attn)).
// Loads torch attn + d_attn, runs TTNN, saves d_scores.

#include "tensordiff/core/serialize.hpp"
#include "tensordiff/core/ttnn_bridge.hpp"

#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>

#include <cstdlib>
#include <filesystem>

using tensordiff::core::load_tensor;
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

static ttnn::WormholeComputeKernelConfig get_compute_cfg() {
    return ttnn::WormholeComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi2,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = true,
    };
}

int main() {
    DeviceGuard guard;
    auto& device = guard.get();

    const char* out_env = std::getenv("TENSORDIFF_OUT_DIR");
    std::filesystem::path out_dir = out_env ? out_env : "tensordiff/outputs/transformer_layer_bwd_torch_rand";

    auto attn = load_tensor((out_dir / "torch_attn_weights.bin").string());
    auto d_attn = load_tensor((out_dir / "torch_d_attn.bin").string());

    auto attn_t = tensordiff::core::to_ttnn(attn, device, false);
    auto d_attn_t = tensordiff::core::to_ttnn(d_attn, device, false);
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    auto dy_y = ttnn::multiply(d_attn_t, attn_t);
    auto sum_dy_y = ttnn::sum(dy_y, -1, true, std::nullopt, get_compute_cfg());
    auto d_scores = ttnn::multiply(attn_t, ttnn::subtract(d_attn_t, sum_dy_y));

    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
    auto d_scores_cpu = tensordiff::core::from_ttnn(d_scores);
    save_tensor(d_scores_cpu, (out_dir / "ttnn_d_scores_manual.bin").string(), "ttnn");

    fmt::print("# wrote: {}\n", out_dir.string());
    return 0;
}
