// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cpu_tensor.hpp"

#include <tt-metalium/bfloat16.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/creation.hpp>
#include <tt-metalium/distributed.hpp>

namespace tensordiff::core {

using TT_Tensor = tt::tt_metal::Tensor;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;

inline TT_Tensor to_ttnn(const CpuTensor& t, MeshDevice& dev, bool row_major = false) {
    if (t.dtype() != DType::kBF16 && t.dtype() != DType::kF32) {
        throw std::runtime_error("to_ttnn: unsupported dtype");
    }
    auto bf16_data = t.to_bf16_vector();
    std::vector<bfloat16> tt_bf16(bf16_data.size());
    for (size_t i = 0; i < bf16_data.size(); ++i) {
        tt_bf16[i] = bfloat16(static_cast<float>(bf16_data[i]));
    }
    ttnn::Shape tt_shape(t.shape());
    auto layout = ttnn::TensorLayout(
        ttnn::DataType::BFLOAT16,
        ttnn::PageConfig(row_major ? ttnn::ROW_MAJOR_LAYOUT : ttnn::TILE_LAYOUT),
        tt::tt_metal::MemoryConfig{}
    );
    auto tensor = TT_Tensor::from_vector(tt_bf16, ttnn::TensorSpec(tt_shape, layout));
    return tensor.to_device(&dev);
}

inline CpuTensor from_ttnn(const TT_Tensor& t) {
    auto cpu_tensor = t.cpu();
    auto shape = cpu_tensor.logical_shape();
    std::vector<uint32_t> dims(shape.rank());
    for (uint32_t i = 0; i < shape.rank(); ++i) {
        dims[i] = shape[i];
    }
    auto data = cpu_tensor.to_vector<bfloat16>();
    std::vector<BFloat16> out(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        out[i] = BFloat16(static_cast<float>(data[i]));
    }
    return CpuTensor::from_bf16(std::move(out), std::move(dims));
}

}  // namespace tensordiff::core
