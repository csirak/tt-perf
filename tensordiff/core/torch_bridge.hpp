// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cpu_tensor.hpp"

#include <torch/torch.h>
#include <cstring>

namespace tensordiff::core {

inline torch::Tensor to_torch(const CpuTensor& t, torch::Dtype dtype = torch::kBFloat16) {
    const auto shape = t.shape();
    std::vector<int64_t> dims(shape.begin(), shape.end());

    if (t.dtype() == DType::kBF16 || t.dtype() == DType::kF32) {
        auto data_f = std::vector<float>(t.numel());
        if (t.dtype() == DType::kBF16) {
            const auto* src = reinterpret_cast<const BFloat16*>(t.raw().data());
            for (size_t i = 0; i < data_f.size(); ++i) {
                data_f[i] = static_cast<float>(src[i]);
            }
        } else {
            const auto* src = reinterpret_cast<const float*>(t.raw().data());
            std::memcpy(data_f.data(), src, data_f.size() * sizeof(float));
        }

        auto tensor = torch::from_blob(data_f.data(), dims, torch::TensorOptions().dtype(torch::kFloat32)).clone();
        return tensor.to(dtype);
    }

    throw std::runtime_error("to_torch: unsupported dtype");
}

inline CpuTensor from_torch(const torch::Tensor& t) {
    auto cpu = t.detach().to(torch::kCPU).contiguous();
    std::vector<uint32_t> shape;
    shape.reserve(cpu.dim());
    for (int64_t i = 0; i < cpu.dim(); ++i) {
        shape.push_back(static_cast<uint32_t>(cpu.size(i)));
    }

    if (cpu.dtype() == torch::kBFloat16) {
        auto flat = cpu.view(-1);
        std::vector<BFloat16> data(flat.numel());
        for (int64_t i = 0; i < flat.numel(); ++i) {
            data[static_cast<size_t>(i)] = BFloat16(flat[i].item<float>());
        }
        return CpuTensor::from_bf16(std::move(data), std::move(shape));
    }
    if (cpu.dtype() == torch::kFloat32) {
        auto flat = cpu.view(-1);
        std::vector<float> data(flat.numel());
        std::memcpy(data.data(), flat.data_ptr<float>(), data.size() * sizeof(float));
        return CpuTensor::from_f32(std::move(data), std::move(shape));
    }

    throw std::runtime_error("from_torch: unsupported dtype");
}

}  // namespace tensordiff::core
