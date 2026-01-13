// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Tensordiff tensor ops test (single readable file).
// - Default build runs TTNN backend.
// - Define TENSORDIFF_TEST_TORCH to run Torch backend.

#include "tensordiff/core/compare.hpp"
#include "tensordiff/core/base_tensor.hpp"

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

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using tensordiff::core::CpuTensor;
using tensordiff::core::DType;
using tensordiff::core::Comparison;
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

static void print_result(const Comparison& r) {
    std::cout << "  " << r.name
              << " | max_abs=" << r.max_abs
              << " mean_abs=" << r.mean_abs
              << " rel_l2=" << r.rel_l2
              << " (numel=" << r.numel << ")\n";
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

    if (const char* dir = std::getenv("TENSORDIFF_OPLOG_DIR")) {
        if (dir[0] != '\0') {
            tensordiff::core::enable_op_recording(dir);
        }
    }

    CpuTensor a({M, K}, DType::kBF16);
    CpuTensor b({K, N}, DType::kBF16);
    a.fill_ramp(-1.0f, 0.01f);
    b.fill_ramp(0.5f, -0.02f);

    auto ref_add = cpu_add(a, b);
    auto ref_sub = cpu_sub(a, b);
    auto ref_mm = cpu_mm(a, b, M, K, N);

#ifdef TENSORDIFF_TEST_TTNN
    std::cout << "# Backend: TTNN\n";
    DeviceGuard guard;
    auto& device = guard.get();

    auto ta = tensordiff::TtnnTensor::from_cpu(a, device);
    auto tb = tensordiff::TtnnTensor::from_cpu(b, device);

    auto add_out = ta.add(tb)->to_cpu();
    auto sub_out = ta.sub(tb)->to_cpu();
    auto mm_out = ta.mm(tb)->to_cpu();

    auto add_cmp = compare_tensors_full("add", add_out, ref_add);
    auto sub_cmp = compare_tensors_full("sub", sub_out, ref_sub);
    auto mm_cmp = compare_tensors_full("mm", mm_out, ref_mm);

    print_result(add_cmp);
    print_result(sub_cmp);
    print_result(mm_cmp);

    ttnn::WormholeComputeKernelConfig cfg{
        .math_fidelity = MathFidelity::HiFi2,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = true,
    };

    auto mm_cfg_out = ta.mm(tb, cfg)->to_cpu();
    auto mm_cfg_cmp = compare_tensors_full("mm_fp32_acc", mm_cfg_out, ref_mm);
    print_result(mm_cfg_cmp);
#endif

#ifdef TENSORDIFF_TEST_TORCH
    std::cout << "# Backend: Torch\n";

    auto ta = tensordiff::TorchTensor::from_cpu(a);
    auto tb = tensordiff::TorchTensor::from_cpu(b);

    auto add_out = ta.add(tb)->to_cpu();
    auto sub_out = ta.sub(tb)->to_cpu();
    auto mm_out = ta.mm(tb)->to_cpu();

    auto add_cmp = compare_tensors_full("add", add_out, ref_add);
    auto sub_cmp = compare_tensors_full("sub", sub_out, ref_sub);
    auto mm_cmp = compare_tensors_full("mm", mm_out, ref_mm);

    print_result(add_cmp);
    print_result(sub_cmp);
    print_result(mm_cmp);
#endif

    tensordiff::core::disable_op_recording();

    return 0;
}
