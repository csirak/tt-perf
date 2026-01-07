// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Static Autograd: Value and Graph
// Automatic differentiation with pre-allocated tensor buffers.
// Values hold pointers to externally-owned tensors for trace compatibility.

#pragma once

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/distributed/api.hpp>

#include <functional>
#include <memory>
#include <vector>
#include <set>

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#endif

namespace static_autograd {

using Tensor = tt::tt_metal::Tensor;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;

// Value: A node in the computation graph
// Points to pre-allocated tensors (does not own them)
struct Value {
    Tensor* data;           // Pointer to forward activation buffer
    Tensor* grad;           // Pointer to gradient buffer (nullptr if !requires_grad)
    bool requires_grad;
    bool grad_initialized = false;  // Track if grad has been written this pass
    std::function<void()> backward_fn;
    std::vector<Value*> parents;

    Value(Tensor* d, Tensor* g, bool req_grad)
        : data(d), grad(g), requires_grad(req_grad) {}

    // Accumulate gradient - overwrites on first call, adds on subsequent
    void accumulate_grad(const Tensor& g) {
#ifdef TRACY_ENABLE
        ZoneScopedN("accumulate_grad");
#endif
        if (!grad) return;
        if (!grad_initialized) {
            *grad = g;
            grad_initialized = true;
        } else {
#ifdef TRACY_ENABLE
            ZoneScopedN("accumulate_grad_add");  // This allocates a new tensor!
#endif
            *grad = ttnn::add(*grad, g);
        }
    }
};

// Graph: Owns all Value nodes and orchestrates backward pass
// Supports persistent mode where nodes are created once and reused.
struct Graph {
    std::vector<std::unique_ptr<Value>> nodes;
    std::vector<Value*> topo_order;  // Cached topological order
    Value* root = nullptr;           // Cached root for backward

    // Create a leaf node (input or parameter)
    Value* leaf(Tensor* data, Tensor* grad, bool requires_grad) {
        nodes.push_back(std::make_unique<Value>(data, grad, requires_grad));
        return nodes.back().get();
    }

    // Create an intermediate node (result of an op)
    Value* node(Tensor* data, Tensor* grad, bool requires_grad = true) {
        nodes.push_back(std::make_unique<Value>(data, grad, requires_grad));
        return nodes.back().get();
    }

    // Reset gradient flags (for persistent graph, call before each backward)
    // No allocation - just marks gradients as uninitialized so next write overwrites
    void zero_grad() {
        for (auto& n : nodes) {
            n->grad_initialized = false;
        }
    }

    // Build and cache topological order (call once after graph construction)
    void build_topo(Value* r) {
        root = r;
        topo_order.clear();
        std::set<Value*> visited;

        std::function<void(Value*)> dfs = [&](Value* v) {
            if (visited.count(v)) return;
            visited.insert(v);
            for (auto* parent : v->parents) {
                dfs(parent);
            }
            topo_order.push_back(v);
        };

        dfs(root);
    }

    // Backward pass using cached topological order
    // If topo_order is empty, builds it first (for compatibility)
    void backward(Value* r) {
#ifdef TRACY_ENABLE
        ZoneScopedN("graph_backward");
#endif
        if (topo_order.empty() || root != r) {
            build_topo(r);
        }

        // Initialize root gradient to ones each call (dL/dL = 1)
        if (root && root->grad) {
            *root->grad = ttnn::ones_like(*root->data);
        }

        // Mark root grad as initialized so it doesn't get overwritten
        root->grad_initialized = true;

        // Backward pass in reverse topological order
        for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
            if ((*it)->backward_fn) {
#ifdef TRACY_ENABLE
                ZoneScopedN("backward_fn");
#endif
                (*it)->backward_fn();
            }
        }
    }
};

// Helper to create tensor on device
inline Tensor make_zeros(
    ttnn::Shape shape,
    MeshDevice& device,
    ttnn::DataType dtype = ttnn::DataType::BFLOAT16) {
    return ttnn::zeros(shape, dtype, ttnn::TILE_LAYOUT, device);
}

inline Tensor make_full(
    ttnn::Shape shape,
    float value,
    MeshDevice& device,
    ttnn::DataType dtype = ttnn::DataType::BFLOAT16) {
    return ttnn::full(shape, value, dtype, ttnn::TILE_LAYOUT, device);
}

inline Tensor make_ones(
    ttnn::Shape shape,
    MeshDevice& device,
    ttnn::DataType dtype = ttnn::DataType::BFLOAT16) {
    return ttnn::ones(shape, dtype, ttnn::TILE_LAYOUT, device);
}

}  // namespace static_autograd
