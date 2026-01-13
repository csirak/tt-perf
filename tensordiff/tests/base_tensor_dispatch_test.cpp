// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// BaseTensor dispatch test (easy-to-read).
// Confirms that BaseTensor::add/sub/mm call the backend-specific impls.

#include "tensordiff/core/base_tensor.hpp"
#include "tensordiff/core/compare.hpp"

#if !defined(TENSORDIFF_TEST_TORCH)
#define TENSORDIFF_TEST_TTNN
#endif

#ifdef TENSORDIFF_TEST_TTNN
#include "tensordiff/ttnn_tensor.hpp"
#include <tt-metalium/distributed.hpp>
#include <ttnn/device.hpp>
#endif

#ifdef TENSORDIFF_TEST_TORCH
#include "tensordiff/torch_tensor.hpp"
#endif

#include <memory>
#include <stdexcept>
#include <vector>

using tensordiff::core::CpuTensor;
using tensordiff::core::DType;
using tensordiff::core::compare_tensors_full;

static CpuTensor cpu_add(const CpuTensor& a, const CpuTensor& b) {
    auto af = tensordiff::core::to_float_vec(a);
    auto bf = tensordiff::core::to_float_vec(b);
    std::vector<float> out(af.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = af[i] + bf[i];
    }
    return CpuTensor::from_f32(std::move(out), a.shape());
}

static CpuTensor cpu_sub(const CpuTensor& a, const CpuTensor& b) {
    auto af = tensordiff::core::to_float_vec(a);
    auto bf = tensordiff::core::to_float_vec(b);
    std::vector<float> out(af.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = af[i] - bf[i];
    }
    return CpuTensor::from_f32(std::move(out), a.shape());
}

static CpuTensor cpu_mm(const CpuTensor& a, const CpuTensor& b, uint32_t m, uint32_t k, uint32_t n) {
    auto af = tensordiff::core::to_float_vec(a);
    auto bf = tensordiff::core::to_float_vec(b);
    std::vector<float> out(static_cast<size_t>(m) * n, 0.0f);
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < k; ++kk) {
                acc += af[static_cast<size_t>(i) * k + kk] * bf[static_cast<size_t>(kk) * n + j];
            }
            out[static_cast<size_t>(i) * n + j] = acc;
        }
    }
    return CpuTensor::from_f32(std::move(out), {m, n});
}

static void require_close(const char* name, const CpuTensor& actual, const CpuTensor& ref, float max_rel_l2) {
    const auto cmp = compare_tensors_full(name, actual, ref);
    if (cmp.rel_l2 > max_rel_l2) {
        throw std::runtime_error(std::string("Mismatch: ") + name);
    }
}

#ifdef TENSORDIFF_TEST_TTNN
struct DeviceGuard {
    std::shared_ptr<tensordiff::TtnnTensor::MeshDevice> device;

    DeviceGuard()
        : device([]() {
            bool use_worker = std::getenv("USE_WORKER_DISPATCH") != nullptr;
            return tensordiff::TtnnTensor::MeshDevice::create_unit_mesh(
                0,
                DEFAULT_L1_SMALL_SIZE,
                0,
                1,
                use_worker ? tt::tt_metal::DispatchCoreConfig{}
                           : tt::tt_metal::DispatchCoreConfig{tt::tt_metal::DispatchCoreType::ETH}
            );
        }()) {}

    ~DeviceGuard() {
        tt::tt_metal::distributed::Finish(device->mesh_command_queue());
        ttnn::close_device(*device);
    }

    tensordiff::TtnnTensor::MeshDevice& get() { return *device; }
};
#endif

int main() {
    constexpr uint32_t M = 32;
    constexpr uint32_t K = 32;
    constexpr uint32_t N = 32;

    CpuTensor a({M, K}, DType::kBF16);
    CpuTensor b({K, N}, DType::kBF16);
    a.fill_ramp(-1.0f, 0.01f);
    b.fill_ramp(0.5f, -0.02f);

    auto ref_add = cpu_add(a, b);
    auto ref_sub = cpu_sub(a, b);
    auto ref_mm = cpu_mm(a, b, M, K, N);

#ifdef TENSORDIFF_TEST_TTNN
    DeviceGuard guard;
    auto& device = guard.get();
    std::unique_ptr<tensordiff::core::BaseTensor> ta =
        std::make_unique<tensordiff::TtnnTensor>(tensordiff::TtnnTensor::from_cpu(a, device));
    std::unique_ptr<tensordiff::core::BaseTensor> tb =
        std::make_unique<tensordiff::TtnnTensor>(tensordiff::TtnnTensor::from_cpu(b, device));
#endif

#ifdef TENSORDIFF_TEST_TORCH
    std::unique_ptr<tensordiff::core::BaseTensor> ta =
        std::make_unique<tensordiff::TorchTensor>(tensordiff::TorchTensor::from_cpu(a));
    std::unique_ptr<tensordiff::core::BaseTensor> tb =
        std::make_unique<tensordiff::TorchTensor>(tensordiff::TorchTensor::from_cpu(b));
#endif

    auto add_out = ta->add(*tb)->to_cpu();
    auto sub_out = ta->sub(*tb)->to_cpu();
    auto mm_out = ta->mm(*tb)->to_cpu();

    require_close("add", add_out, ref_add, 0.01f);
    require_close("sub", sub_out, ref_sub, 0.01f);
    require_close("mm", mm_out, ref_mm, 0.01f);

    return 0;
}
