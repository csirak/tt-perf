#pragma once

#include "common.hpp"

namespace static_autograd {

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

// =============================================================================
// LayerNorm: Layer normalization with learnable gamma/beta
// Input: [B, S, D] - normalizes over last dimension
// =============================================================================

} // namespace static_autograd
