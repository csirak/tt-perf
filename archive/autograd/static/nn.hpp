// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Static Autograd Neural Network Layers
// Layers own their parameter and gradient buffers.
// Forward pass takes a Graph& and returns Value* for automatic differentiation.

#pragma once

#include "ops.hpp"
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>
#include <ttnn/operations/eltwise/ternary/ternary_composite.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <ttnn/operations/matmul/device/matmul_op.hpp>
#include <ttnn/operations/copy/typecast/typecast.hpp>
#include <cmath>
#include <vector>

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#endif

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
#ifdef TRACY_ENABLE
        ZoneScopedN("Linear_forward");
#endif
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

// =============================================================================
// LayerNorm: Layer normalization with learnable gamma/beta
// Input: [B, S, D] - normalizes over last dimension
// =============================================================================
struct LayerNorm {
    // Parameters
    Tensor gamma;       // [1, 1, D]
    Tensor beta;        // [1, 1, D]
    Tensor d_gamma;
    Tensor d_beta;

    // Forward buffers
    Tensor out;
    Tensor d_out;
    Tensor x_norm;      // cached normalized input for backward
    Tensor rstd;        // cached 1/sqrt(var+eps) for backward

    uint32_t dim;
    float eps;

    LayerNorm(uint32_t batch, uint32_t seq, uint32_t d, float epsilon, MeshDevice& dev)
        : gamma(make_full(ttnn::Shape({1, 1, d}), 1.0f, dev)),
          beta(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          d_gamma(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          d_beta(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          out(make_zeros(ttnn::Shape({batch, seq, d}), dev)),
          d_out(make_zeros(ttnn::Shape({batch, seq, d}), dev)),
          x_norm(make_zeros(ttnn::Shape({batch, seq, d}), dev)),
          rstd(make_zeros(ttnn::Shape({batch, seq, 1}), dev)),
          dim(d),
          eps(epsilon) {}

    Value* forward(Graph& g, Value* x) {
#ifdef TRACY_ENABLE
        ZoneScopedN("LayerNorm_forward");
#endif
        auto* gam = g.leaf(&gamma, &d_gamma, true);
        auto* bet = g.leaf(&beta, &d_beta, true);

        // mean = mean(x, dim=-1, keepdim=True)
        auto mean_val = ttnn::mean(*x->data, -1, true);
        // x_centered = x - mean
        auto x_centered = ttnn::subtract(*x->data, mean_val);
        // var = mean(x_centered^2)
        auto var = ttnn::mean(ttnn::multiply(x_centered, x_centered), -1, true);
        // rstd = 1/sqrt(var + eps)
        rstd = ttnn::rsqrt(ttnn::add(var, eps), true);
        // x_norm = x_centered * rstd
        x_norm = ttnn::multiply(x_centered, rstd);
        // out = gamma * x_norm + beta
        out = ttnn::add(ttnn::multiply(gamma, x_norm), beta);

        auto* v = g.node(&out, &d_out);
        v->parents = {x, gam, bet};

        v->backward_fn = [x, gam, bet, this, v]() {
            if (!v->grad) return;
            const auto& dout = *v->grad;

            // d_beta = sum(dout, dims=[0,1])
            if (bet->requires_grad) {
                auto sum0 = ttnn::sum(dout, 0, true);
                bet->accumulate_grad(ttnn::sum(sum0, 1, true));
            }

            // d_gamma = sum(dout * x_norm, dims=[0,1])
            if (gam->requires_grad) {
                auto d_out_x_norm = ttnn::multiply(dout, x_norm);
                auto sum0_g = ttnn::sum(d_out_x_norm, 0, true);
                gam->accumulate_grad(ttnn::sum(sum0_g, 1, true));
            }

            // d_x = rstd * (d_x_norm - mean(d_x_norm) - x_norm * mean(d_x_norm * x_norm))
            if (x->requires_grad) {
                auto d_x_norm = ttnn::multiply(dout, gamma);
                auto mean_d_x_norm = ttnn::mean(d_x_norm, -1, true);
                auto d_x_norm_x_norm = ttnn::multiply(d_x_norm, x_norm);
                auto mean_d_x_norm_x_norm = ttnn::mean(d_x_norm_x_norm, -1, true);
                auto diff1 = ttnn::subtract(d_x_norm, mean_d_x_norm);
                auto term2 = ttnn::multiply(x_norm, mean_d_x_norm_x_norm);
                auto diff2 = ttnn::subtract(diff1, term2);
                x->accumulate_grad(ttnn::multiply(rstd, diff2));
            }
        };

        return v;
    }

    void sgd_step(float lr) {
        gamma = ttnn::subtract(gamma, ttnn::multiply(d_gamma, lr));
        beta = ttnn::subtract(beta, ttnn::multiply(d_beta, lr));
    }
};

// =============================================================================
// LayerNormBFP8: Layer normalization that outputs BFP8 directly
// Uses BF16 for intermediate computations (stability), BFP8 for final output.
// Eliminates typecast overhead when feeding into BFP8 linear layers.
// =============================================================================
struct LayerNormBFP8 {
    // Parameters (BF16 for gradient stability)
    Tensor gamma;       // [1, 1, D]
    Tensor beta;        // [1, 1, D]
    Tensor d_gamma;
    Tensor d_beta;

    // Forward buffers
    Tensor out;         // BFP8 output
    Tensor d_out;       // BF16 gradient
    Tensor x_norm;      // BF16 cached normalized input for backward
    Tensor rstd;        // BF16 cached 1/sqrt(var+eps) for backward

    uint32_t dim;
    float eps;

    LayerNormBFP8(uint32_t batch, uint32_t seq, uint32_t d, float epsilon, MeshDevice& dev)
        : gamma(make_full(ttnn::Shape({1, 1, d}), 1.0f, dev)),
          beta(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          d_gamma(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          d_beta(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          out(make_zeros_bfp8(ttnn::Shape({batch, seq, d}), dev)),  // BFP8 output
          d_out(make_zeros(ttnn::Shape({batch, seq, d}), dev)),
          x_norm(make_zeros(ttnn::Shape({batch, seq, d}), dev)),
          rstd(make_zeros(ttnn::Shape({batch, seq, 1}), dev)),
          dim(d),
          eps(epsilon) {}

    Value* forward(Graph& g, Value* x) {
#ifdef TRACY_ENABLE
        ZoneScopedN("LayerNormBFP8_forward");
#endif
        auto* gam = g.leaf(&gamma, &d_gamma, true);
        auto* bet = g.leaf(&beta, &d_beta, true);

        // All intermediate computations in BF16 for numerical stability
        // mean = mean(x, dim=-1, keepdim=True)
        auto mean_val = ttnn::mean(*x->data, -1, true);
        // x_centered = x - mean
        auto x_centered = ttnn::subtract(*x->data, mean_val);
        // var = mean(x_centered^2)
        auto var = ttnn::mean(ttnn::multiply(x_centered, x_centered), -1, true);
        // rstd = 1/sqrt(var + eps)
        rstd = ttnn::rsqrt(ttnn::add(var, eps), true);
        // x_norm = x_centered * rstd
        x_norm = ttnn::multiply(x_centered, rstd);
        // scaled = gamma * x_norm
        auto scaled = ttnn::multiply(gamma, x_norm);
        // out = scaled + beta, with output cast to BFP8 using preallocated buffer
        // API: ttnn::add(lhs, rhs, output_dtype, memory_config, output, ...)
        ttnn::add(scaled, beta, ttnn::DataType::BFLOAT8_B, std::nullopt, out);

        auto* v = g.node(&out, &d_out);
        v->parents = {x, gam, bet};

        v->backward_fn = [x, gam, bet, this, v]() {
#ifdef TRACY_ENABLE
            ZoneScopedN("LayerNormBFP8_backward");
#endif
            if (!v->grad) return;

            // Convert BFP8 gradient to BF16 for stable backward pass
            Tensor dout;
            {
#ifdef TRACY_ENABLE
                ZoneScopedN("ln_typecast_dout");
#endif
                dout = ttnn::typecast(*v->grad, ttnn::DataType::BFLOAT16);
            }

            // d_beta = sum(dout, dims=[0,1])
            if (bet->requires_grad) {
#ifdef TRACY_ENABLE
                ZoneScopedN("ln_d_beta");
#endif
                auto sum0 = ttnn::sum(dout, 0, true);
                bet->accumulate_grad(ttnn::sum(sum0, 1, true));
            }

            // d_gamma = sum(dout * x_norm, dims=[0,1])
            if (gam->requires_grad) {
#ifdef TRACY_ENABLE
                ZoneScopedN("ln_d_gamma");
#endif
                auto d_out_x_norm = ttnn::multiply(dout, x_norm);
                auto sum0_g = ttnn::sum(d_out_x_norm, 0, true);
                gam->accumulate_grad(ttnn::sum(sum0_g, 1, true));
            }

            // d_x = rstd * (d_x_norm - mean(d_x_norm) - x_norm * mean(d_x_norm * x_norm))
            if (x->requires_grad) {
#ifdef TRACY_ENABLE
                ZoneScopedN("ln_d_x");
#endif
                auto d_x_norm = ttnn::multiply(dout, gamma);
                auto mean_d_x_norm = ttnn::mean(d_x_norm, -1, true);
                auto d_x_norm_x_norm = ttnn::multiply(d_x_norm, x_norm);
                auto mean_d_x_norm_x_norm = ttnn::mean(d_x_norm_x_norm, -1, true);
                auto diff1 = ttnn::subtract(d_x_norm, mean_d_x_norm);
                auto term2 = ttnn::multiply(x_norm, mean_d_x_norm_x_norm);
                auto diff2 = ttnn::subtract(diff1, term2);
                x->accumulate_grad(ttnn::multiply(rstd, diff2));
            }
        };

        return v;
    }

    void sgd_step(float lr) {
        gamma = ttnn::subtract(gamma, ttnn::multiply(d_gamma, lr));
        beta = ttnn::subtract(beta, ttnn::multiply(d_beta, lr));
    }
};

// =============================================================================
// Linear3D: Linear layer for 3D tensors [B, S, D]
// Matches transformer usage where input is [batch, seq, dim]
// =============================================================================
struct Linear3D {
    Tensor weight;      // [out_dim, in_dim]
    Tensor bias;        // [1, 1, out_dim]
    Tensor d_weight;
    Tensor d_bias;

    Tensor out;         // [B, S, out_dim]
    Tensor d_out;

    uint32_t in_dim;
    uint32_t out_dim;

    Linear3D(uint32_t batch, uint32_t seq, uint32_t in_d, uint32_t out_d, float init, MeshDevice& dev)
        : weight(make_full(ttnn::Shape({out_d, in_d}), init, dev)),
          bias(make_zeros(ttnn::Shape({1, 1, out_d}), dev)),
          d_weight(make_zeros(ttnn::Shape({out_d, in_d}), dev)),
          d_bias(make_zeros(ttnn::Shape({1, 1, out_d}), dev)),
          out(make_zeros(ttnn::Shape({batch, seq, out_d}), dev)),
          d_out(make_zeros(ttnn::Shape({batch, seq, out_d}), dev)),
          in_dim(in_d),
          out_dim(out_d) {}

    Value* forward(Graph& g, Value* x) {
#ifdef TRACY_ENABLE
        ZoneScopedN("Linear3D_forward");
#endif
        auto* w = g.leaf(&weight, &d_weight, true);
        auto* b = g.leaf(&bias, &d_bias, true);

        // out = x @ weight.T + bias
        out = ttnn::add(ttnn::matmul(*x->data, weight, false, true), bias);

        auto* v = g.node(&out, &d_out);
        v->parents = {x, w, b};

        v->backward_fn = [x, w, b, this, v]() {
            if (!v->grad) return;
            const auto& dout = *v->grad;

            // d_input = dout @ weight
            if (x->requires_grad) {
                x->accumulate_grad(ttnn::matmul(dout, weight));
            }
            // d_weight = dout.T @ x (summed over batch and seq)
            if (w->requires_grad) {
                // For 3D: need to reshape or handle carefully
                // dout: [B, S, out], x: [B, S, in]
                // d_weight = sum over B,S of dout[b,s,:].T @ x[b,s,:]
                // Use matmul with transpose_a=true
                w->accumulate_grad(ttnn::matmul(dout, *x->data, true, false));
            }
            // d_bias = sum(dout, dims=[0,1])
            if (b->requires_grad) {
                auto sum0 = ttnn::sum(dout, 0, true);
                b->accumulate_grad(ttnn::sum(sum0, 1, true));
            }
        };

        return v;
    }

    void sgd_step(float lr) {
        weight = ttnn::subtract(weight, ttnn::multiply(d_weight, lr));
        bias = ttnn::subtract(bias, ttnn::multiply(d_bias, lr));
    }
};

// =============================================================================
// FFN: Feed-Forward Network with GELU activation
// Linear -> GELU -> Linear
// =============================================================================
struct FFN {
    Linear3D w1;        // [dim, ffn_dim]
    Linear3D w2;        // [ffn_dim, dim]

    // GELU intermediate buffers
    Tensor gelu_out;
    Tensor d_gelu_out;

    FFN(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t ffn_dim, float init, MeshDevice& dev)
        : w1(batch, seq, dim, ffn_dim, init, dev),
          w2(batch, seq, ffn_dim, dim, init, dev),
          gelu_out(make_zeros(ttnn::Shape({batch, seq, ffn_dim}), dev)),
          d_gelu_out(make_zeros(ttnn::Shape({batch, seq, ffn_dim}), dev)) {}

    Value* forward(Graph& g, Value* x) {
#ifdef TRACY_ENABLE
        ZoneScopedN("FFN_forward");
#endif
        auto* h = w1.forward(g, x);
        h = gelu(g, h, &gelu_out, &d_gelu_out);
        return w2.forward(g, h);
    }

    void sgd_step(float lr) {
        w1.sgd_step(lr);
        w2.sgd_step(lr);
    }
};

// =============================================================================
// Linear3DBFP8: BFP8 Linear layer with fused matmul+bias (ttnn::linear)
// Uses BFP8 weights, bias, and activations. BF16 gradients for stability.
// Expects BFP8 input (from LayerNormBFP8) - no typecast needed.
// Key insight: Mixed-type add (BFP8 + BF16) is 4x slower than same-type.
// =============================================================================
struct Linear3DBFP8 {
    // BFP8 weights and bias (all same type for fast fused linear)
    Tensor weight;      // BFP8 [out_dim, in_dim]
    Tensor bias;        // BFP8 [1, 1, out_dim] - must be BFP8 for fast add!
    // BF16 gradients for numerical stability
    Tensor d_weight;    // BF16
    Tensor d_bias;      // BF16
    // BF16 master copies for SGD (accumulate in BF16, convert to BFP8)
    Tensor weight_bf16; // BF16 copy for gradient accumulation
    Tensor bias_bf16;   // BF16 copy for gradient accumulation

    Tensor out;         // BFP8 output
    Tensor d_out;       // BF16 gradient

    uint32_t in_dim;
    uint32_t out_dim;

    // Optimized compute config - using HiFi2 for stability (full 7-bit precision)
    // Can switch to LoFi for 2x throughput if training is stable
    static ttnn::WormholeComputeKernelConfig get_compute_config() {
        return ttnn::WormholeComputeKernelConfig{
            .math_fidelity = MathFidelity::HiFi2,  // Full precision for training stability
            .math_approx_mode = false,
            .fp32_dest_acc_en = false,
            .packer_l1_acc = true,
        };
    }

    Linear3DBFP8(uint32_t batch, uint32_t seq, uint32_t in_d, uint32_t out_d, float init, MeshDevice& dev)
        : weight(make_full_bfp8(ttnn::Shape({out_d, in_d}), init, dev)),
          bias(make_zeros_bfp8(ttnn::Shape({1, 1, out_d}), dev)),  // BFP8 bias!
          d_weight(make_zeros(ttnn::Shape({out_d, in_d}), dev)),
          d_bias(make_zeros(ttnn::Shape({1, 1, out_d}), dev)),
          weight_bf16(make_full(ttnn::Shape({out_d, in_d}), init, dev)),
          bias_bf16(make_zeros(ttnn::Shape({1, 1, out_d}), dev)),
          out(make_zeros_bfp8(ttnn::Shape({batch, seq, out_d}), dev)),
          d_out(make_zeros(ttnn::Shape({batch, seq, out_d}), dev)),
          in_dim(in_d),
          out_dim(out_d) {}

    Value* forward(Graph& g, Value* x) {
#ifdef TRACY_ENABLE
        ZoneScopedN("Linear3DBFP8_forward");
#endif
        auto* w = g.leaf(&weight, &d_weight, true);
        auto* b = g.leaf(&bias, &d_bias, true);

        // Fused BFP8 linear: x @ weight.T + bias (all BFP8, no mixed-type add!)
        out = ttnn::linear(*x->data, weight, bias,
                           /*transpose_a=*/false, /*transpose_b=*/true,
                           /*memory_config=*/std::nullopt,
                           /*dtype=*/std::nullopt,
                           /*program_config=*/std::nullopt,
                           /*activation=*/std::nullopt,
                           /*compute_kernel_config=*/get_compute_config());

        auto* v = g.node(&out, &d_out);
        v->parents = {x, w, b};

        v->backward_fn = [x, w, b, this, v]() {
#ifdef TRACY_ENABLE
            ZoneScopedN("Linear3DBFP8_backward");
#endif
            if (!v->grad) return;

            // Convert gradient to BF16 for stability
            Tensor dout_bf16;
            {
#ifdef TRACY_ENABLE
                ZoneScopedN("typecast_dout_bf16");
#endif
                dout_bf16 = ttnn::typecast(*v->grad, ttnn::DataType::BFLOAT16);
            }

            // d_input = dout @ weight (need BF16 weight for backward)
            if (x->requires_grad) {
#ifdef TRACY_ENABLE
                ZoneScopedN("d_input_matmul");
#endif
                x->accumulate_grad(ttnn::matmul(dout_bf16, weight_bf16));
            }
            // d_weight = dout.T @ x
            if (w->requires_grad) {
#ifdef TRACY_ENABLE
                ZoneScopedN("d_weight_compute");
#endif
                auto x_bf16 = ttnn::typecast(*x->data, ttnn::DataType::BFLOAT16);
                w->accumulate_grad(ttnn::matmul(dout_bf16, x_bf16, true, false));
            }
            // d_bias = sum(dout, dims=[0,1])
            if (b->requires_grad) {
#ifdef TRACY_ENABLE
                ZoneScopedN("d_bias_sum");
#endif
                auto sum0 = ttnn::sum(dout_bf16, 0, true);
                b->accumulate_grad(ttnn::sum(sum0, 1, true));
            }
        };

        return v;
    }

    void sgd_step(float lr) {
#ifdef TRACY_ENABLE
        ZoneScopedN("Linear3DBFP8_sgd_step");
#endif
        // Update BF16 master copies (broadcasting handles 3D gradient → 2D weight)
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("sgd_weight_update");
#endif
            weight_bf16 = ttnn::subtract(weight_bf16, ttnn::multiply(d_weight, lr));
        }
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("sgd_bias_update");
#endif
            bias_bf16 = ttnn::subtract(bias_bf16, ttnn::multiply(d_bias, lr));
        }
        // Cast to pre-allocated BFP8 buffers - DEVICE SIDE
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("sgd_typecast_weight");
#endif
            ttnn::typecast(weight_bf16, ttnn::DataType::BFLOAT8_B, std::nullopt, weight);
        }
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("sgd_typecast_bias");
#endif
            ttnn::typecast(bias_bf16, ttnn::DataType::BFLOAT8_B, std::nullopt, bias);
        }
    }
};

// =============================================================================
// FFNBFP8: Feed-Forward Network with BFP8 matmuls
// Linear (BFP8) -> GELU -> Linear (BFP8)
// =============================================================================
struct FFNBFP8 {
    Linear3DBFP8 w1;    // [dim, ffn_dim]
    Linear3DBFP8 w2;    // [ffn_dim, dim]

    // GELU intermediate buffers (BFP8 for compute, BF16 for grad)
    Tensor gelu_out;    // BFP8
    Tensor d_gelu_out;  // BF16

    FFNBFP8(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t ffn_dim, float init, MeshDevice& dev)
        : w1(batch, seq, dim, ffn_dim, init, dev),
          w2(batch, seq, ffn_dim, dim, init, dev),
          gelu_out(make_zeros_bfp8(ttnn::Shape({batch, seq, ffn_dim}), dev)),
          d_gelu_out(make_zeros(ttnn::Shape({batch, seq, ffn_dim}), dev)) {}

    Value* forward(Graph& g, Value* x) {
#ifdef TRACY_ENABLE
        ZoneScopedN("FFNBFP8_forward");
#endif
        auto* h = w1.forward(g, x);
        // GELU operates on BFP8
        h = gelu(g, h, &gelu_out, &d_gelu_out);
        return w2.forward(g, h);
    }

    void sgd_step(float lr) {
        w1.sgd_step(lr);
        w2.sgd_step(lr);
    }
};

// =============================================================================
// PersistentTransformerLayer: Single transformer layer for GPT-2
// Architecture: LN1 -> Attention -> Add -> LN2 -> FFN -> Add
// Uses simplified attention (no reshape/transpose for now)
// =============================================================================
struct PersistentTransformerLayer {
    // Modules
    LayerNorm ln1;
    LayerNorm ln2;
    Linear3D wq, wk, wv, wo;
    FFN ffn;

    // Attention config
    float scale;
    uint32_t batch, seq, dim, heads, head_dim;

    // Forward buffers
    Tensor q_proj, k_proj, v_proj;          // [B, S, D]
    Tensor d_q_proj, d_k_proj, d_v_proj;
    Tensor attn_scores;                      // [B, S, S] simplified (single head view)
    Tensor d_attn_scores;
    Tensor attn_weights;                     // [B, S, S]
    Tensor d_attn_weights;
    Tensor attn_out;                         // [B, S, D]
    Tensor d_attn_out;
    Tensor attn_proj;                        // [B, S, D]
    Tensor d_attn_proj;
    Tensor residual1;                        // [B, S, D]
    Tensor d_residual1;
    Tensor ffn_out;                          // [B, S, D]
    Tensor d_ffn_out;
    Tensor output;                           // [B, S, D]
    Tensor d_output;

    // Causal mask
    Tensor causal_mask;                      // [1, S, S]

    PersistentTransformerLayer(uint32_t b, uint32_t s, uint32_t d, uint32_t h,
                                uint32_t ffn_mult, float init, MeshDevice& dev)
        : ln1(b, s, d, 1e-5f, dev),
          ln2(b, s, d, 1e-5f, dev),
          wq(b, s, d, d, init, dev),
          wk(b, s, d, d, init, dev),
          wv(b, s, d, d, init, dev),
          wo(b, s, d, d, init, dev),
          ffn(b, s, d, d * ffn_mult, init, dev),
          scale(1.0f / std::sqrt(static_cast<float>(d / h))),
          batch(b), seq(s), dim(d), heads(h), head_dim(d / h),
          q_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          k_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          v_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_q_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_k_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_v_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          attn_scores(make_zeros(ttnn::Shape({b, s, s}), dev)),
          d_attn_scores(make_zeros(ttnn::Shape({b, s, s}), dev)),
          attn_weights(make_zeros(ttnn::Shape({b, s, s}), dev)),
          d_attn_weights(make_zeros(ttnn::Shape({b, s, s}), dev)),
          attn_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_attn_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          attn_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_attn_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          residual1(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_residual1(make_zeros(ttnn::Shape({b, s, d}), dev)),
          ffn_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_ffn_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          output(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_output(make_zeros(ttnn::Shape({b, s, d}), dev)),
          causal_mask(create_causal_mask(s, dev)) {}

    static Tensor create_causal_mask(uint32_t s, MeshDevice& dev) {
        auto mask = make_full(ttnn::Shape({1, s, s}), -1e9f, dev);
        return ttnn::triu(mask, 1);
    }

    // Simplified attention (no multi-head reshape for initial testing)
    // Real implementation would need reshape [B,S,D] -> [B,H,S,D/H]
    Value* forward(Graph& g, Value* x) {
#ifdef TRACY_ENABLE
        ZoneScopedN("TransformerLayer_forward");
#endif
        // LN1
        auto* ln1_v = ln1.forward(g, x);

        // QKV projections
        auto* q_v = wq.forward(g, ln1_v);
        auto* k_v = wk.forward(g, ln1_v);
        auto* v_v = wv.forward(g, ln1_v);

        // Simplified attention: scores = Q @ K.T * scale
        attn_scores = ttnn::multiply(ttnn::matmul(*q_v->data, *k_v->data, false, true), scale);

        // Apply causal mask
        auto scores_masked = ttnn::add(attn_scores, causal_mask);

        // Softmax
        auto x_max = ttnn::max(scores_masked, -1, true);
        auto x_centered = ttnn::subtract(scores_masked, x_max);
        attn_weights = ttnn::softmax(x_centered, -1);

        // attn_out = attn_weights @ V
        attn_out = ttnn::matmul(attn_weights, *v_v->data);

        // Create node for attention output
        auto* attn_v = g.node(&attn_out, &d_attn_out);
        attn_v->parents = {q_v, k_v, v_v};

        // Attention backward (simplified - captures q,k,v for gradient computation)
        attn_v->backward_fn = [q_v, k_v, v_v, this, attn_v]() {
            if (!attn_v->grad) return;
            const auto& dout = *attn_v->grad;

            // d_attn_weights = dout @ V.T
            auto d_attn = ttnn::matmul(dout, *v_v->data, false, true);

            // d_V = attn_weights.T @ dout
            if (v_v->requires_grad) {
                v_v->accumulate_grad(ttnn::matmul(attn_weights, dout, true, false));
            }

            // Softmax backward
            auto dy_y = ttnn::multiply(d_attn, attn_weights);
            auto sum_dy_y = ttnn::sum(dy_y, -1, true);
            auto d_scores = ttnn::multiply(attn_weights, ttnn::subtract(d_attn, sum_dy_y));
            auto d_scores_scaled = ttnn::multiply(d_scores, scale);

            // d_Q = d_scores_scaled @ K
            if (q_v->requires_grad) {
                q_v->accumulate_grad(ttnn::matmul(d_scores_scaled, *k_v->data));
            }
            // d_K = d_scores_scaled.T @ Q
            if (k_v->requires_grad) {
                k_v->accumulate_grad(ttnn::matmul(d_scores_scaled, *q_v->data, true, false));
            }
        };

        // Output projection
        auto* attn_proj_v = wo.forward(g, attn_v);

        // Residual 1
        auto* res1_v = add(g, x, attn_proj_v, &residual1, &d_residual1);

        // LN2
        auto* ln2_v = ln2.forward(g, res1_v);

        // FFN
        auto* ffn_v = ffn.forward(g, ln2_v);

        // Residual 2
        return add(g, res1_v, ffn_v, &output, &d_output);
    }

    void sgd_step(float lr) {
        ln1.sgd_step(lr);
        ln2.sgd_step(lr);
        wq.sgd_step(lr);
        wk.sgd_step(lr);
        wv.sgd_step(lr);
        wo.sgd_step(lr);
        ffn.sgd_step(lr);
    }
};

// =============================================================================
// PersistentTransformerLayerBFP8: Transformer layer with BFP8 FFN
// Same as PersistentTransformerLayer but uses FFNBFP8 for higher throughput.
// ln2 uses LayerNormBFP8 to output BFP8 directly, eliminating typecast overhead.
// =============================================================================
struct PersistentTransformerLayerBFP8 {
    // Modules (attention stays BF16, FFN uses BFP8)
    LayerNorm ln1;            // BF16 output (attention is BF16)
    LayerNormBFP8 ln2;        // BFP8 output (feeds directly into BFP8 FFN)
    Linear3D wq, wk, wv, wo;  // Attention projections stay BF16
    FFNBFP8 ffn;              // BFP8 FFN for high throughput

    // Attention config
    float scale;
    uint32_t batch, seq, dim, heads, head_dim;

    // Forward buffers
    Tensor q_proj, k_proj, v_proj;
    Tensor d_q_proj, d_k_proj, d_v_proj;
    Tensor attn_scores, d_attn_scores;
    Tensor attn_weights, d_attn_weights;
    Tensor attn_out, d_attn_out;
    Tensor attn_proj, d_attn_proj;
    Tensor residual1, d_residual1;
    Tensor ffn_out, d_ffn_out;
    Tensor output, d_output;
    Tensor causal_mask;

    PersistentTransformerLayerBFP8(uint32_t b, uint32_t s, uint32_t d, uint32_t h,
                                    uint32_t ffn_mult, float init, MeshDevice& dev)
        : ln1(b, s, d, 1e-5f, dev),
          ln2(b, s, d, 1e-5f, dev),
          wq(b, s, d, d, init, dev),
          wk(b, s, d, d, init, dev),
          wv(b, s, d, d, init, dev),
          wo(b, s, d, d, init, dev),
          ffn(b, s, d, d * ffn_mult, init, dev),  // BFP8 FFN
          scale(1.0f / std::sqrt(static_cast<float>(d / h))),
          batch(b), seq(s), dim(d), heads(h), head_dim(d / h),
          q_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          k_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          v_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_q_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_k_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_v_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          attn_scores(make_zeros(ttnn::Shape({b, s, s}), dev)),
          d_attn_scores(make_zeros(ttnn::Shape({b, s, s}), dev)),
          attn_weights(make_zeros(ttnn::Shape({b, s, s}), dev)),
          d_attn_weights(make_zeros(ttnn::Shape({b, s, s}), dev)),
          attn_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_attn_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          attn_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_attn_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          residual1(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_residual1(make_zeros(ttnn::Shape({b, s, d}), dev)),
          ffn_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_ffn_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          output(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_output(make_zeros(ttnn::Shape({b, s, d}), dev)),
          causal_mask(PersistentTransformerLayer::create_causal_mask(s, dev)) {}

    Value* forward(Graph& g, Value* x) {
#ifdef TRACY_ENABLE
        ZoneScopedN("TransformerLayerBFP8_forward");
#endif
        // LN1
        auto* ln1_v = ln1.forward(g, x);

        // QKV projections
        auto* q_v = wq.forward(g, ln1_v);
        auto* k_v = wk.forward(g, ln1_v);
        auto* v_v = wv.forward(g, ln1_v);

        // Simplified attention: scores = Q @ K.T * scale
        attn_scores = ttnn::multiply(ttnn::matmul(*q_v->data, *k_v->data, false, true), scale);

        // Apply causal mask
        auto scores_masked = ttnn::add(attn_scores, causal_mask);

        // Softmax
        auto x_max = ttnn::max(scores_masked, -1, true);
        auto x_centered = ttnn::subtract(scores_masked, x_max);
        attn_weights = ttnn::softmax(x_centered, -1);

        // attn_out = attn_weights @ V
        attn_out = ttnn::matmul(attn_weights, *v_v->data);

        // Create node for attention output
        auto* attn_v = g.node(&attn_out, &d_attn_out);
        attn_v->parents = {q_v, k_v, v_v};

        // Attention backward
        attn_v->backward_fn = [q_v, k_v, v_v, this, attn_v]() {
            if (!attn_v->grad) return;
            const auto& dout = *attn_v->grad;

            auto d_attn = ttnn::matmul(dout, *v_v->data, false, true);
            if (v_v->requires_grad) {
                v_v->accumulate_grad(ttnn::matmul(attn_weights, dout, true, false));
            }

            auto dy_y = ttnn::multiply(d_attn, attn_weights);
            auto sum_dy_y = ttnn::sum(dy_y, -1, true);
            auto d_scores = ttnn::multiply(attn_weights, ttnn::subtract(d_attn, sum_dy_y));
            auto d_scores_scaled = ttnn::multiply(d_scores, scale);

            if (q_v->requires_grad) {
                q_v->accumulate_grad(ttnn::matmul(d_scores_scaled, *k_v->data));
            }
            if (k_v->requires_grad) {
                k_v->accumulate_grad(ttnn::matmul(d_scores_scaled, *q_v->data, true, false));
            }
        };

        // Output projection
        auto* attn_proj_v = wo.forward(g, attn_v);

        // Residual 1
        auto* res1_v = add(g, x, attn_proj_v, &residual1, &d_residual1);

        // LN2
        auto* ln2_v = ln2.forward(g, res1_v);

        // BFP8 FFN
        auto* ffn_v = ffn.forward(g, ln2_v);

        // Residual 2 - need to convert BFP8 FFN output back to BF16
        // The add op will handle mixed precision
        return add(g, res1_v, ffn_v, &output, &d_output);
    }

    void sgd_step(float lr) {
        ln1.sgd_step(lr);
        ln2.sgd_step(lr);
        wq.sgd_step(lr);
        wk.sgd_step(lr);
        wv.sgd_step(lr);
        wo.sgd_step(lr);
        ffn.sgd_step(lr);
    }
};

// =============================================================================
// PersistentGPT2BFP8: GPT-2 with BFP8 FFN layers for high throughput
// =============================================================================
template<size_t N>
struct PersistentGPT2BFP8 {
    static_assert(N >= 1, "Must have at least 1 layer");

    std::vector<PersistentTransformerLayerBFP8> layers;
    Linear3D output_proj;

    Tensor input, d_input;
    Tensor target;
    Tensor loss, d_loss, diff;

    Graph graph;
    Value* input_node = nullptr;
    Value* loss_node = nullptr;
    bool built = false;

    uint32_t batch, seq, dim, vocab;
    float lr;

    PersistentGPT2BFP8(uint32_t b, uint32_t s, uint32_t d, uint32_t h, uint32_t ffn_mult,
                       uint32_t v, float learning_rate, MeshDevice& dev)
        : output_proj(b, s, d, v, 0.02f, dev),
          input(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_input(make_zeros(ttnn::Shape({b, s, d}), dev)),
          target(make_zeros(ttnn::Shape({b, s, v}), dev)),
          loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          d_loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          diff(make_zeros(ttnn::Shape({b, s, v}), dev)),
          batch(b), seq(s), dim(d), vocab(v),
          lr(learning_rate)
    {
        layers.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            layers.emplace_back(b, s, d, h, ffn_mult, 0.02f, dev);
        }
    }

    void build() {
        input_node = graph.leaf(&input, &d_input, false);

        Value* h = input_node;
        for (size_t i = 0; i < N; ++i) {
            h = layers[i].forward(graph, h);
        }

        auto* logits = output_proj.forward(graph, h);
        loss_node = mse(graph, logits, &target, &loss, &d_loss, &diff);
        graph.build_topo(loss_node);
        built = true;
    }

    void train_step() {
        if (!built) build();
        graph.zero_grad();
        graph.backward(loss_node);
        for (size_t i = 0; i < N; ++i) {
            layers[i].sgd_step(lr);
        }
        output_proj.sgd_step(lr);
    }
};

// =============================================================================
// PersistentGPT2: Full GPT-2 style model with N transformer layers
// Uses PersistentMLP pattern: build graph once, execute many times
// =============================================================================
template<size_t N>
struct PersistentGPT2 {
    static_assert(N >= 1, "Must have at least 1 layer");

    // Transformer layers
    std::vector<PersistentTransformerLayer> layers;

    // Output projection
    Linear3D output_proj;

    // Input/output buffers
    Tensor input;           // [B, S, D]
    Tensor d_input;
    Tensor target;          // [B, S, vocab]

    // Loss buffers
    Tensor loss;
    Tensor d_loss;
    Tensor diff;

    // Persistent graph
    Graph graph;
    Value* input_node = nullptr;
    Value* loss_node = nullptr;
    bool built = false;

    uint32_t batch, seq, dim, vocab;
    float lr;

    PersistentGPT2(uint32_t b, uint32_t s, uint32_t d, uint32_t h, uint32_t ffn_mult,
                   uint32_t v, float learning_rate, MeshDevice& dev)
        : output_proj(b, s, d, v, 0.02f, dev),
          input(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_input(make_zeros(ttnn::Shape({b, s, d}), dev)),
          target(make_zeros(ttnn::Shape({b, s, v}), dev)),
          loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          d_loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          diff(make_zeros(ttnn::Shape({b, s, v}), dev)),
          batch(b), seq(s), dim(d), vocab(v),
          lr(learning_rate)
    {
        layers.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            layers.emplace_back(b, s, d, h, ffn_mult, 0.02f, dev);
        }
    }

    // Build computation graph (call once)
    void build() {
        input_node = graph.leaf(&input, &d_input, false);

        // Forward through transformer layers
        Value* h = input_node;
        for (size_t i = 0; i < N; ++i) {
            h = layers[i].forward(graph, h);
        }

        // Output projection
        auto* logits = output_proj.forward(graph, h);

        // MSE loss
        loss_node = mse(graph, logits, &target, &loss, &d_loss, &diff);
        graph.build_topo(loss_node);
        built = true;
    }

    // Execute forward (overwrites same buffers)
    void execute_forward() {
        // This would need to manually re-execute all ops
        // For now, rely on the graph's backward_fn capturing the right tensors
        // The key insight is that tensors are class members that get overwritten
    }

    // Full training step
    void train_step() {
        if (!built) build();

        // Forward - executed implicitly during graph construction
        // For persistent execution, we'd need execute_forward()

        // Backward
        graph.zero_grad();
        graph.backward(loss_node);

        // SGD
        for (size_t i = 0; i < N; ++i) {
            layers[i].sgd_step(lr);
        }
        output_proj.sgd_step(lr);
    }

    float get_loss(MeshDevice* dev) {
        tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        return static_cast<float>(loss.cpu().to_vector<bfloat16>()[0]);
    }
};

}  // namespace static_autograd
