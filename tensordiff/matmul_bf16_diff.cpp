// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// BF16 matmul diff: CPU reference vs TTNN matmul.

#include "tensordiff/core/cpu_tensor.hpp"
#include "tensordiff/core/serialize.hpp"
#include "tensordiff/core/compare.hpp"
#include "tensordiff/core/ttnn_bridge.hpp"

#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>

#include <cstdlib>
#include <filesystem>
#include <random>

using tensordiff::core::BFloat16;
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

static CpuTensor cpu_matmul_bf16(const CpuTensor& a, const CpuTensor& b) {
    if (a.dtype() != DType::kBF16 || b.dtype() != DType::kBF16) {
        throw std::runtime_error("cpu_matmul_bf16: inputs must be BF16");
    }
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::runtime_error("cpu_matmul_bf16: inputs must be 2D");
    }
    const uint32_t m = a.shape()[0];
    const uint32_t k = a.shape()[1];
    const uint32_t kb = b.shape()[0];
    const uint32_t n = b.shape()[1];
    if (k != kb) {
        throw std::runtime_error("cpu_matmul_bf16: shape mismatch");
    }

    const auto* a_data = reinterpret_cast<const BFloat16*>(a.raw().data());
    const auto* b_data = reinterpret_cast<const BFloat16*>(b.raw().data());

    std::vector<BFloat16> out(static_cast<size_t>(m) * n);
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (uint32_t kk = 0; kk < k; ++kk) {
                float av = static_cast<float>(a_data[static_cast<size_t>(i) * k + kk]);
                float bv = static_cast<float>(b_data[static_cast<size_t>(kk) * n + j]);
                sum += av * bv;
            }
            out[static_cast<size_t>(i) * n + j] = BFloat16(sum);
        }
    }

    return CpuTensor::from_bf16(std::move(out), {m, n});
}

static void fill_uniform_bf16(CpuTensor& t, float min_val, float max_val, std::mt19937& rng) {
    if (t.dtype() != DType::kBF16) {
        throw std::runtime_error("fill_uniform_bf16: dtype must be BF16");
    }
    std::uniform_real_distribution<float> dist(min_val, max_val);
    auto* dst = reinterpret_cast<BFloat16*>(t.raw().data());
    for (size_t i = 0; i < t.numel(); ++i) {
        dst[i] = BFloat16(dist(rng));
    }
}

int main() {
    DeviceGuard guard;
    auto& device = guard.get();

    const char* out_env = std::getenv("TENSORDIFF_OUT_DIR");
    std::filesystem::path out_dir = out_env ? out_env : "tensordiff/outputs/matmul";
    std::filesystem::create_directories(out_dir);

    uint32_t M = 64;
    uint32_t K = 64;
    uint32_t N = 64;
    if (const char* env = std::getenv("TDF_M")) M = static_cast<uint32_t>(std::stoul(env));
    if (const char* env = std::getenv("TDF_K")) K = static_cast<uint32_t>(std::stoul(env));
    if (const char* env = std::getenv("TDF_N")) N = static_cast<uint32_t>(std::stoul(env));

    if ((M % 32) || (K % 32) || (N % 32)) {
        fmt::print("# ERROR: M/K/N must be multiples of 32 (got M={}, K={}, N={})\n", M, K, N);
        return 1;
    }

    uint32_t seed = 0;
    if (const char* env = std::getenv("TDF_SEED")) seed = static_cast<uint32_t>(std::stoul(env));

    float init_min = -0.5f;
    float init_max = 0.5f;
    if (const char* env = std::getenv("TDF_MIN")) init_min = std::stof(env);
    if (const char* env = std::getenv("TDF_MAX")) init_max = std::stof(env);

    CpuTensor a({M, K}, DType::kBF16);
    CpuTensor b({K, N}, DType::kBF16);
    std::mt19937 rng(seed);
    fill_uniform_bf16(a, init_min, init_max, rng);
    fill_uniform_bf16(b, init_min, init_max, rng);

    auto ref = cpu_matmul_bf16(a, b);

    auto tt_a = tensordiff::core::to_ttnn(a, device, false);
    auto tt_b = tensordiff::core::to_ttnn(b, device, false);
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    auto tt_out = ttnn::matmul(tt_a, tt_b);
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    auto cmp = tensordiff::core::from_ttnn(tt_out);

    save_tensor(ref, (out_dir / "matmul_ref.bin").string(), "cpu");
    save_tensor(cmp, (out_dir / "matmul_ttnn.bin").string(), "ttnn");

    auto result = compare_tensors("matmul_bf16", ref, cmp);
    fmt::print("# matmul_bf16: {} elements\n", result.numel);
    fmt::print("# max_abs={:.6f} mean_abs={:.6f} rel_l2={:.6f}\n",
               result.max_abs, result.mean_abs, result.rel_l2);
    fmt::print("# shape: M={}, K={}, N={}, seed={}, init=[{}, {}]\n",
               M, K, N, seed, init_min, init_max);
    fmt::print("# wrote: {}\n", out_dir.string());

    return 0;
}
