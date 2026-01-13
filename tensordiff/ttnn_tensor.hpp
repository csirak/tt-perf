// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "core/base_tensor.hpp"
#include "core/ttnn_bridge.hpp"

#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>

#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tensordiff {

class TtnnTensor final : public core::BaseTensor {
public:
    using TT_Tensor = core::TT_Tensor;
    using MeshDevice = core::MeshDevice;
    using core::BaseTensor::mm;

    TtnnTensor(TT_Tensor tensor, MeshDevice* device)
        : tensor_(std::move(tensor)), device_(device) {
        if (!device_) {
            throw std::runtime_error("TtnnTensor: device is null");
        }
    }

    static TtnnTensor from_cpu(const core::CpuTensor& cpu, MeshDevice& device, bool row_major = false) {
        return TtnnTensor(core::to_ttnn(cpu, device, row_major), &device);
    }

    core::CpuTensor to_cpu() const override {
        return core::from_ttnn(tensor_);
    }

    std::vector<uint32_t> shape() const override {
        auto s = tensor_.logical_shape();
        std::vector<uint32_t> dims(s.rank());
        for (uint32_t i = 0; i < s.rank(); ++i) {
            dims[i] = s[i];
        }
        return dims;
    }

    core::DType dtype() const override {
        return core::DType::kBF16;
    }

    std::string backend() const override {
        return "ttnn";
    }

protected:
    std::unique_ptr<core::BaseTensor> add_impl(const core::BaseTensor& other) const override {
        const auto* o = dynamic_cast<const TtnnTensor*>(&other);
        if (!o) {
            throw std::runtime_error("TtnnTensor::add expects TtnnTensor");
        }
        auto out = ttnn::add(tensor_, o->tensor_);
        return std::make_unique<TtnnTensor>(std::move(out), device_);
    }

    std::unique_ptr<core::BaseTensor> sub_impl(const core::BaseTensor& other) const override {
        const auto* o = dynamic_cast<const TtnnTensor*>(&other);
        if (!o) {
            throw std::runtime_error("TtnnTensor::sub expects TtnnTensor");
        }
        auto out = ttnn::subtract(tensor_, o->tensor_);
        return std::make_unique<TtnnTensor>(std::move(out), device_);
    }

    std::unique_ptr<core::BaseTensor> mm_impl(const core::BaseTensor& other) const override {
        const auto* o = dynamic_cast<const TtnnTensor*>(&other);
        if (!o) {
            throw std::runtime_error("TtnnTensor::mm expects TtnnTensor");
        }
        auto out = ttnn::matmul(tensor_, o->tensor_, false, false);
        return std::make_unique<TtnnTensor>(std::move(out), device_);
    }

public:

    std::unique_ptr<TtnnTensor> mm(const TtnnTensor& other,
                                   const std::optional<ttnn::WormholeComputeKernelConfig>& config) const {
        TT_Tensor out;
        if (config.has_value()) {
            out = ttnn::matmul(
                tensor_,
                other.tensor_,
                false,
                false,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                *config);
        } else {
            out = ttnn::matmul(tensor_, other.tensor_, false, false);
        }
        auto result = std::make_unique<TtnnTensor>(std::move(out), device_);
        constexpr std::string_view op_name = "mm";
        std::string opts;
        if (config.has_value()) {
            std::ostringstream oss;
            oss << "math_fidelity=" << static_cast<int>(config->math_fidelity)
                << ",approx=" << (config->math_approx_mode ? "1" : "0")
                << ",fp32_acc=" << (config->fp32_dest_acc_en ? "1" : "0")
                << ",packer_l1=" << (config->packer_l1_acc ? "1" : "0");
            opts = oss.str();
        }
        record_op(op_name, core::OpType::Binary, {this, &other}, *result, opts);
        return result;
    }

    const TT_Tensor& raw() const { return tensor_; }

private:
    TT_Tensor tensor_;
    MeshDevice* device_ = nullptr;
};

}  // namespace tensordiff
