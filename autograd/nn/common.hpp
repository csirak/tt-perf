// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Static Autograd Neural Network Layers
// Layers own their parameter and gradient buffers.
// Forward pass takes a Graph& and returns Value* for automatic differentiation.

#pragma once

#include "../ops.hpp"
#include "../tensor_group.hpp"
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>
#include <ttnn/operations/eltwise/ternary/ternary_composite.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/matmul/device/matmul_op.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <ttnn/operations/rand/rand.hpp>
#include <cmath>
#include <vector>
#include <cstdlib>

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#endif

namespace static_autograd {

// Import TypecastCache from static_mxp namespace
using static_mxp::TypecastCache;
using static_mxp::WeightTypecastCache;
using static_mxp::TensorGroup;

enum class InitKind {
    Constant,
    Zeros,
    Randn,
};

inline uint32_t g_seed_counter = 1;

inline Tensor make_randn(const ttnn::Shape& shape, float std, MeshDevice& device) {
    float range = std::sqrt(3.0f) * std;
    return ttnn::rand(shape, device, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT,
                      ttnn::types::DRAM_MEMORY_CONFIG, -range, range, g_seed_counter++);
}

inline Tensor make_init_tensor(const ttnn::Shape& shape, float init, MeshDevice& device, InitKind kind) {
    switch (kind) {
        case InitKind::Zeros:
            return make_zeros(shape, device);
        case InitKind::Randn:
            return make_randn(shape, init, device);
        case InitKind::Constant:
        default:
            return make_full(shape, init, device);
    }
}

inline Tensor make_embedding_weight(const ttnn::Shape& shape, float std, MeshDevice& device) {
    float range = std::sqrt(3.0f) * std;
    auto tiled = ttnn::rand(shape, device, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT,
                            ttnn::types::DRAM_MEMORY_CONFIG, -range, range, g_seed_counter++);
    return ttnn::untilize(tiled);
}

inline Tensor make_indices(const std::vector<uint32_t>& indices, const ttnn::Shape& shape, MeshDevice& device) {
    auto tensor_layout = ttnn::TensorLayout(
        ttnn::DataType::UINT32,
        ttnn::PageConfig(ttnn::ROW_MAJOR_LAYOUT),
        tt::tt_metal::MemoryConfig{}
    );
    auto tensor = Tensor::from_vector(indices, ttnn::TensorSpec(shape, tensor_layout));
    return tensor.to_device(&device);
}

} // namespace static_autograd
