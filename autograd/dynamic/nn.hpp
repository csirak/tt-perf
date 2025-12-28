// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Neural Network Layers for TTNN Autograd

#pragma once

#include "autograd.hpp"
#include "ops.hpp"

namespace autograd {

// Linear layer: Y = X @ W.T + bias
struct Linear {
    ValuePtr weight;  // [out_features, in_features]
    ValuePtr bias;    // [1, out_features]

    Linear(uint32_t in_features, uint32_t out_features, MeshDevice& device)
        : weight(make_random(ttnn::Shape({out_features, in_features}), device, -0.1f, 0.1f, true, "weight")),
          bias(make_zeros(ttnn::Shape({1, out_features}), device, true, "bias")) {}

    ValuePtr forward(ValuePtr x) {
        // x: [batch, in_features], weight: [out, in] -> w_t: [in, out]
        auto w_t = std::make_shared<Value>(
            ttnn::transpose(weight->data, -2, -1),
            weight->requires_grad, "weight_t", std::vector<ValuePtr>{weight}
        );
        w_t->backward_fn = [this, w_t]() {
            if (!w_t->grad.has_value()) return;
            weight->accumulate_grad(ttnn::transpose(w_t->grad.value(), -2, -1));
        };
        return add(mm(x, w_t), bias);
    }

    std::vector<ValuePtr> parameters() { return {weight, bias}; }
};

// 2-layer MLP with ReLU activation
struct MLP {
    Linear layer1;
    Linear layer2;

    MLP(uint32_t in_dim, uint32_t hidden_dim, uint32_t out_dim, MeshDevice& device)
        : layer1(in_dim, hidden_dim, device),
          layer2(hidden_dim, out_dim, device) {}

    ValuePtr forward(ValuePtr x) {
        auto h = relu(layer1.forward(x));
        return layer2.forward(h);
    }

    std::vector<ValuePtr> parameters() {
        return {layer1.weight, layer1.bias, layer2.weight, layer2.bias};
    }
};

}  // namespace autograd
