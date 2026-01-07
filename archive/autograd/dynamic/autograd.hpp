// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Minimal Autograd Engine for TTNN
// Step 1: Basic tensor wrapper with gradient tracking

#pragma once

#include <ttnn/device.hpp>
#include <ttnn/types.hpp>
#include <ttnn/tensor/shape/shape.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/core/core.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/functions.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/distributed/api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_config.hpp>

#include <functional>
#include <memory>
#include <vector>
#include <optional>
#include <set>

namespace autograd {

using namespace ttnn;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;

// Forward declaration
struct Value;
using ValuePtr = std::shared_ptr<Value>;

// Value: A tensor with optional gradient tracking
// This is the core building block of our autograd system
struct Value {
    // The actual data tensor (on device)
    tt::tt_metal::Tensor data;

    // Gradient tensor (same shape as data, lazily allocated)
    std::optional<tt::tt_metal::Tensor> grad;

    // Whether this value requires gradient computation
    bool requires_grad = false;

    // Backward function: called during backprop to compute gradients
    // This captures the inputs and the operation that created this value
    std::function<void()> backward_fn;

    // Parents in the computation graph (for topological sort during backward)
    std::vector<ValuePtr> parents;

    // Name for debugging
    std::string name;

    // Constructor from existing tensor
    Value(tt::tt_metal::Tensor t, bool requires_grad_ = false, std::string name_ = "",
          std::vector<ValuePtr> parents_ = {})
        : data(std::move(t)), requires_grad(requires_grad_),
          parents(std::move(parents_)), name(std::move(name_)) {}

    // Get shape
    const tt::tt_metal::Shape& shape() const {
        return data.logical_shape();
    }

    // Initialize gradient to zeros (same shape as data)
    void zero_grad(MeshDevice& device) {
        if (requires_grad) {
            grad = ttnn::zeros(shape(), DataType::BFLOAT16, TILE_LAYOUT, device);
        }
    }

    // Accumulate gradient (grad += incoming_grad)
    // Important for nodes with multiple consumers
    void accumulate_grad(const tt::tt_metal::Tensor& incoming_grad) {
        if (!requires_grad) return;
        grad = grad.has_value()
            ? ttnn::add(grad.value(), incoming_grad)
            : incoming_grad;
    }
};

// Create a Value from shape (initialized with zeros or ones)
inline ValuePtr make_zeros(ttnn::Shape shape, MeshDevice& device, bool requires_grad = false, const std::string& name = "") {
    auto t = ttnn::zeros(shape, DataType::BFLOAT16, TILE_LAYOUT, device);
    return std::make_shared<Value>(std::move(t), requires_grad, name);
}

inline ValuePtr make_ones(ttnn::Shape shape, MeshDevice& device, bool requires_grad = false, const std::string& name = "") {
    auto t = ttnn::ones(shape, DataType::BFLOAT16, TILE_LAYOUT, device);
    return std::make_shared<Value>(std::move(t), requires_grad, name);
}

inline ValuePtr make_random(ttnn::Shape shape, MeshDevice& device, float low = -1.0f, float high = 1.0f,
                            bool requires_grad = false, const std::string& name = "") {
    auto t_host = ttnn::random::uniform(bfloat16(low), bfloat16(high), shape, TILE_LAYOUT);
    auto t = ttnn::to_device(t_host, &device, MemoryConfig{});
    return std::make_shared<Value>(std::move(t), requires_grad, name);
}

// Create a Value from existing tensor
inline ValuePtr from_tensor(tt::tt_metal::Tensor t, bool requires_grad = false, const std::string& name = "") {
    return std::make_shared<Value>(std::move(t), requires_grad, name);
}

// Backward pass: topological sort + call backward_fn in reverse order
inline void backward(ValuePtr root, MeshDevice& device) {
    // Build topological order via DFS
    std::vector<ValuePtr> topo_order;
    std::set<Value*> visited;

    std::function<void(ValuePtr)> build_topo = [&](ValuePtr v) {
        if (visited.count(v.get())) return;
        visited.insert(v.get());
        for (auto& parent : v->parents) {
            build_topo(parent);
        }
        topo_order.push_back(v);
    };

    build_topo(root);

    // Initialize root gradient to ones (dL/dL = 1)
    root->grad = ttnn::ones(root->shape(), DataType::BFLOAT16, TILE_LAYOUT, device);

    // Backward pass in reverse topological order
    for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
        if ((*it)->backward_fn) {
            (*it)->backward_fn();
        }
    }
}

}  // namespace autograd
