// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Static Autograd Neural Network Layers
// Layers own their parameter and gradient buffers.
// Forward pass takes a Graph& and returns Value* for automatic differentiation.

#pragma once

#include "ops.hpp"

namespace static_autograd {

// Linear layer: y = x @ weight.T + bias
// Owns: weight, bias, d_weight, d_bias, output buffer
struct Linear {
    // Parameters
    Tensor weight;      // [out_features, in_features]
    Tensor bias;        // [1, out_features]

    // Gradients
    Tensor d_weight;
    Tensor d_bias;

    // Output buffer
    Tensor out;
    Tensor d_out;

    // Intermediate for matmul (x @ weight.T)
    Tensor mm_out;
    Tensor d_mm_out;

    // Transposed weight for forward (weight.T = [in, out])
    Tensor weight_t;
    Tensor d_weight_t;

    uint32_t in_features;
    uint32_t out_features;
    uint32_t batch_size;

    Linear(uint32_t batch, uint32_t in_f, uint32_t out_f, float init, MeshDevice& dev)
        : weight(make_full(ttnn::Shape({out_f, in_f}), init, dev)),
          bias(make_zeros(ttnn::Shape({1, out_f}), dev)),
          d_weight(make_zeros(ttnn::Shape({out_f, in_f}), dev)),
          d_bias(make_zeros(ttnn::Shape({1, out_f}), dev)),
          out(make_zeros(ttnn::Shape({batch, out_f}), dev)),
          d_out(make_zeros(ttnn::Shape({batch, out_f}), dev)),
          mm_out(make_zeros(ttnn::Shape({batch, out_f}), dev)),
          d_mm_out(make_zeros(ttnn::Shape({batch, out_f}), dev)),
          weight_t(make_zeros(ttnn::Shape({in_f, out_f}), dev)),
          d_weight_t(make_zeros(ttnn::Shape({in_f, out_f}), dev)),
          in_features(in_f),
          out_features(out_f),
          batch_size(batch) {}

    // Forward: returns Value* pointing to output buffer
    Value* forward(Graph& g, Value* x) {
        // Create value nodes for parameters
        auto* w = g.leaf(&weight, &d_weight, true);
        auto* b = g.leaf(&bias, &d_bias, true);

        // Transpose weight for matmul: x @ weight.T
        // We use matmul with transpose_b=true instead of explicit transpose
        // But we need to track gradients properly

        // y = x @ weight.T + bias
        // Using matmul with transpose_b flag
        mm_out = ttnn::matmul(*x->data, weight, false, true);

        auto* mm_v = g.node(&mm_out, &d_mm_out);
        mm_v->parents = {x, w};

        mm_v->backward_fn = [x, w, this, mm_v]() {
            if (!mm_v->grad) return;
            const auto& dout = *mm_v->grad;

            // dx = dout @ weight (no transpose - weight is [out, in])
            if (x->requires_grad) {
                x->accumulate_grad(ttnn::matmul(dout, weight));
            }
            // dweight = dout.T @ x
            if (w->requires_grad) {
                w->accumulate_grad(ttnn::matmul(dout, *x->data, true, false));
            }
        };

        // Add bias
        return add(g, mm_v, b, &out, &d_out);
    }

    // SGD update
    void sgd_step(float lr) {
        weight = ttnn::subtract(weight, ttnn::multiply(d_weight, lr));
        bias = ttnn::subtract(bias, ttnn::multiply(d_bias, lr));
    }

    // Zero gradients
    void zero_grad() {
        d_weight = ttnn::zeros_like(d_weight);
        d_bias = ttnn::zeros_like(d_bias);
    }
};

// =============================================================================
// LinearReLU: Fused Linear + ReLU using ttnn::linear with activation parameter
// Fuses matmul + bias + relu into a single kernel for better performance.
// =============================================================================
struct LinearReLU {
    // Parameters
    Tensor weight;      // [out_features, in_features]
    Tensor bias;        // [1, out_features]

    // Gradients
    Tensor d_weight;
    Tensor d_bias;

    // Output buffer (post-activation)
    Tensor out;
    Tensor d_out;

    // ReLU mask for backward
    Tensor relu_mask;

    uint32_t in_features;
    uint32_t out_features;
    uint32_t batch_size;

    LinearReLU(uint32_t batch, uint32_t in_f, uint32_t out_f, float init, MeshDevice& dev)
        : weight(make_full(ttnn::Shape({out_f, in_f}), init, dev)),
          bias(make_zeros(ttnn::Shape({1, out_f}), dev)),
          d_weight(make_zeros(ttnn::Shape({out_f, in_f}), dev)),
          d_bias(make_zeros(ttnn::Shape({1, out_f}), dev)),
          out(make_zeros(ttnn::Shape({batch, out_f}), dev)),
          d_out(make_zeros(ttnn::Shape({batch, out_f}), dev)),
          relu_mask(make_zeros(ttnn::Shape({batch, out_f}), dev)),
          in_features(in_f),
          out_features(out_f),
          batch_size(batch) {}

    Value* forward(Graph& g, Value* x) {
        auto* w = g.leaf(&weight, &d_weight, true);
        auto* b = g.leaf(&bias, &d_bias, true);

        // Fused: y = relu(x @ weight.T + bias) in single kernel
        out = ttnn::linear(*x->data, weight, bias, false, true,
                           std::nullopt, std::nullopt, std::nullopt, "relu");

        // Compute ReLU mask for backward (out > 0 since ReLU already applied)
        relu_mask = ttnn::gtz(out);

        auto* v = g.node(&out, &d_out);
        v->parents = {x, w, b};

        v->backward_fn = [x, w, b, this, v]() {
            if (!v->grad) return;
            const auto& dout = *v->grad;

            // Apply ReLU mask to incoming gradient
            auto dout_masked = ttnn::multiply(dout, relu_mask);

            // dx = dout_masked @ weight (weight is [out, in])
            if (x->requires_grad) {
                x->accumulate_grad(ttnn::matmul(dout_masked, weight));
            }
            // dweight = dout_masked.T @ x
            if (w->requires_grad) {
                w->accumulate_grad(ttnn::matmul(dout_masked, *x->data, true, false));
            }
            // dbias = sum(dout_masked, dim=0)
            if (b->requires_grad) {
                b->accumulate_grad(ttnn::sum(dout_masked, 0, true));
            }
        };

        return v;
    }

    void sgd_step(float lr) {
        weight = ttnn::subtract(weight, ttnn::multiply(d_weight, lr));
        bias = ttnn::subtract(bias, ttnn::multiply(d_bias, lr));
    }

    void zero_grad() {
        d_weight = ttnn::zeros_like(d_weight);
        d_bias = ttnn::zeros_like(d_bias);
    }
};

// Simple 2-layer MLP with ReLU for testing (unfused baseline)
struct MLP {
    Linear layer1;
    Linear layer2;

    // ReLU intermediate buffers
    Tensor relu_out;
    Tensor d_relu_out;
    Tensor relu_mask;

    MLP(uint32_t batch, uint32_t in_dim, uint32_t hidden_dim, uint32_t out_dim,
        float init, MeshDevice& dev)
        : layer1(batch, in_dim, hidden_dim, init, dev),
          layer2(batch, hidden_dim, out_dim, init, dev),
          relu_out(make_zeros(ttnn::Shape({batch, hidden_dim}), dev)),
          d_relu_out(make_zeros(ttnn::Shape({batch, hidden_dim}), dev)),
          relu_mask(make_zeros(ttnn::Shape({batch, hidden_dim}), dev)) {}

    Value* forward(Graph& g, Value* x) {
        auto* h = layer1.forward(g, x);
        h = relu(g, h, &relu_out, &d_relu_out, &relu_mask);
        return layer2.forward(g, h);
    }

    void sgd_step(float lr) {
        layer1.sgd_step(lr);
        layer2.sgd_step(lr);
    }

    void zero_grad() {
        layer1.zero_grad();
        layer2.zero_grad();
    }
};

// =============================================================================
// FusedMLP: 2-layer MLP using LinearReLU for fused hidden layer
// Faster than MLP due to kernel fusion (matmul+bias+relu in one kernel).
// =============================================================================
struct FusedMLP {
    LinearReLU layer1;    // Hidden layer with fused ReLU
    Linear layer2;        // Output layer (no activation)

    FusedMLP(uint32_t batch, uint32_t in_dim, uint32_t hidden_dim, uint32_t out_dim,
             float init, MeshDevice& dev)
        : layer1(batch, in_dim, hidden_dim, init, dev),
          layer2(batch, hidden_dim, out_dim, init, dev) {}

    Value* forward(Graph& g, Value* x) {
        auto* h = layer1.forward(g, x);  // Fused matmul+bias+relu
        return layer2.forward(g, h);     // Regular linear
    }

    void sgd_step(float lr) {
        layer1.sgd_step(lr);
        layer2.sgd_step(lr);
    }

    void zero_grad() {
        layer1.zero_grad();
        layer2.zero_grad();
    }
};

// =============================================================================
// Persistent MLP: Graph built once, reused across iterations
// Supports TTNN trace API by separating graph construction from execution.
// =============================================================================
struct PersistentMLP {
    // Layers
    Linear layer1;
    Linear layer2;

    // ReLU buffers
    Tensor relu_out;
    Tensor d_relu_out;
    Tensor relu_mask;

    // Input/output buffers (owned by this struct for persistence)
    Tensor input;
    Tensor d_input;
    Tensor target;

    // Loss buffers
    Tensor loss;
    Tensor d_loss;
    Tensor diff;

    // Persistent graph
    Graph graph;
    Value* input_node = nullptr;
    Value* loss_node = nullptr;
    bool built = false;

    float lr;

    PersistentMLP(uint32_t batch, uint32_t dim, float learning_rate, MeshDevice& dev)
        : layer1(batch, dim, dim, 0.01f, dev),
          layer2(batch, dim, dim, 0.01f, dev),
          relu_out(make_zeros(ttnn::Shape({batch, dim}), dev)),
          d_relu_out(make_zeros(ttnn::Shape({batch, dim}), dev)),
          relu_mask(make_zeros(ttnn::Shape({batch, dim}), dev)),
          input(make_zeros(ttnn::Shape({batch, dim}), dev)),
          d_input(make_zeros(ttnn::Shape({batch, dim}), dev)),
          target(make_full(ttnn::Shape({batch, dim}), 0.5f, dev)),
          loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          d_loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          diff(make_zeros(ttnn::Shape({batch, dim}), dev)),
          lr(learning_rate) {}

    // Build the computation graph (call once)
    void build() {
        input_node = graph.leaf(&input, &d_input, false);
        auto* h = layer1.forward(graph, input_node);
        h = relu(graph, h, &relu_out, &d_relu_out, &relu_mask);
        auto* pred = layer2.forward(graph, h);
        loss_node = mse(graph, pred, &target, &loss, &d_loss, &diff);
        graph.build_topo(loss_node);
        built = true;
    }

    // Execute forward pass only (reuses existing graph nodes)
    void execute_forward() {
        // Re-run forward computation - writes to same buffers
        // layer1: mm_out = input @ weight.T, out = mm_out + bias
        layer1.mm_out = ttnn::matmul(input, layer1.weight, false, true);
        layer1.out = ttnn::add(layer1.mm_out, layer1.bias);

        // ReLU
        relu_mask = ttnn::gtz(layer1.out);
        relu_out = ttnn::relu(layer1.out);

        // layer2: mm_out = relu_out @ weight.T, out = mm_out + bias
        layer2.mm_out = ttnn::matmul(relu_out, layer2.weight, false, true);
        layer2.out = ttnn::add(layer2.mm_out, layer2.bias);

        // MSE loss
        diff = ttnn::subtract(layer2.out, target);
        auto sq = ttnn::multiply(diff, diff);
        loss = ttnn::mean(sq, std::nullopt, true);
    }

    // Full training step
    void train_step() {
        if (!built) build();

        // Forward
        execute_forward();

        // Backward - graph.zero_grad() resets grad_initialized flags (no allocation)
        graph.zero_grad();
        graph.backward(loss_node);

        // SGD
        layer1.sgd_step(lr);
        layer2.sgd_step(lr);
    }
};

}  // namespace static_autograd
