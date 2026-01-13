// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Load LayerNorm inputs/params from TDF1, run TTNN layer_norm, save outputs.

#include "tensordiff/core/serialize.hpp"
#include "tensordiff/core/ttnn_bridge.hpp"

#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>

#include <cstdlib>
#include <filesystem>
#include <cstring>

using tensordiff::core::CpuTensor;
using tensordiff::core::DType;
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

static CpuTensor expand_to_1x1d(const CpuTensor& t) {
    if (t.shape().size() != 1) {
        return t;
    }
    uint32_t d = t.shape()[0];
    CpuTensor out({1, 1, d}, t.dtype());
    std::memcpy(out.raw().data(), t.raw().data(), t.numel() * tensordiff::core::dtype_size(t.dtype()));
    return out;
}

static CpuTensor squeeze_batch(const CpuTensor& t) {
    if (t.shape().size() != 3 || t.shape()[0] != 1) {
        return t;
    }
    uint32_t s = t.shape()[1];
    uint32_t d = t.shape()[2];
    CpuTensor out({s, d}, t.dtype());
    const size_t bytes = static_cast<size_t>(s) * d * tensordiff::core::dtype_size(t.dtype());
    std::memcpy(out.raw().data(), t.raw().data(), bytes);
    return out;
}

int main() {
    DeviceGuard guard;
    auto& device = guard.get();

    const char* out_env = std::getenv("TENSORDIFF_OUT_DIR");
    std::filesystem::path out_dir = out_env ? out_env : "tensordiff/outputs/layernorm_torch_rand";

    double eps = 1e-5;
    if (const char* env = std::getenv("TDF_EPS")) eps = std::stod(env);

    auto x = load_tensor((out_dir / "torch_ln_in.bin").string());
    auto gamma = load_tensor((out_dir / "torch_ln_gamma.bin").string());
    auto beta = load_tensor((out_dir / "torch_ln_beta.bin").string());

    auto gamma3 = expand_to_1x1d(gamma);
    auto beta3 = expand_to_1x1d(beta);

    auto tt_x = tensordiff::core::to_ttnn(x, device, false);
    auto tt_gamma = tensordiff::core::to_ttnn(gamma3, device, false);
    auto tt_beta = tensordiff::core::to_ttnn(beta3, device, false);
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    auto tt_out = ttnn::layer_norm(tt_x, static_cast<float>(eps), tt_gamma, tt_beta,
                                   std::nullopt, std::nullopt, std::nullopt, get_compute_cfg());
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    auto x_round = tensordiff::core::from_ttnn(tt_x);
    auto out_cpu = tensordiff::core::from_ttnn(tt_out);
    auto out_2d = squeeze_batch(out_cpu);
    auto x_2d = squeeze_batch(x_round);

    save_tensor(x_round, (out_dir / "ttnn_ln_in.bin").string(), "ttnn");
    save_tensor(out_cpu, (out_dir / "ttnn_ln_out.bin").string(), "ttnn");
    save_tensor(x_2d, (out_dir / "ttnn_ln_in_2d.bin").string(), "ttnn");
    save_tensor(out_2d, (out_dir / "ttnn_ln_out_2d.bin").string(), "ttnn");

    save_tensor(tensordiff::core::from_ttnn(tt_gamma), (out_dir / "ttnn_ln_gamma.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(tt_beta), (out_dir / "ttnn_ln_beta.bin").string(), "ttnn");

    fmt::print("# wrote: {}\n", out_dir.string());
    return 0;
}
