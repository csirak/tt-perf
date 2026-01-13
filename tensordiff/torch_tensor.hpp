// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "core/base_tensor.hpp"
#include "core/torch_bridge.hpp"

#include <torch/torch.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensordiff {

class TorchTensor final : public core::BaseTensor {
public:
    explicit TorchTensor(torch::Tensor tensor)
        : tensor_(std::move(tensor)) {}

    static TorchTensor from_cpu(const core::CpuTensor& cpu, torch::Dtype dtype = torch::kBFloat16) {
        return TorchTensor(core::to_torch(cpu, dtype));
    }

    core::CpuTensor to_cpu() const override {
        return core::from_torch(tensor_);
    }

    std::vector<uint32_t> shape() const override {
        std::vector<uint32_t> dims;
        dims.reserve(static_cast<size_t>(tensor_.dim()));
        for (int64_t i = 0; i < tensor_.dim(); ++i) {
            dims.push_back(static_cast<uint32_t>(tensor_.size(i)));
        }
        return dims;
    }

    core::DType dtype() const override {
        if (tensor_.dtype() == torch::kFloat32) {
            return core::DType::kF32;
        }
        return core::DType::kBF16;
    }

    std::string backend() const override {
        return "torch";
    }

protected:
    std::unique_ptr<core::BaseTensor> add_impl(const core::BaseTensor& other) const override {
        const auto* o = dynamic_cast<const TorchTensor*>(&other);
        if (!o) {
            throw std::runtime_error("TorchTensor::add expects TorchTensor");
        }
        return std::make_unique<TorchTensor>(tensor_ + o->tensor_);
    }

    std::unique_ptr<core::BaseTensor> sub_impl(const core::BaseTensor& other) const override {
        const auto* o = dynamic_cast<const TorchTensor*>(&other);
        if (!o) {
            throw std::runtime_error("TorchTensor::sub expects TorchTensor");
        }
        return std::make_unique<TorchTensor>(tensor_ - o->tensor_);
    }

    std::unique_ptr<core::BaseTensor> mm_impl(const core::BaseTensor& other) const override {
        const auto* o = dynamic_cast<const TorchTensor*>(&other);
        if (!o) {
            throw std::runtime_error("TorchTensor::mm expects TorchTensor");
        }
        return std::make_unique<TorchTensor>(torch::matmul(tensor_, o->tensor_));
    }

public:

    const torch::Tensor& raw() const { return tensor_; }

private:
    torch::Tensor tensor_;
};

}  // namespace tensordiff
