// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Traced Neural Network Layers
// Layers with pre-allocated gradient buffers for trace compatibility.

#pragma once

#include "ops.hpp"
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>
#include <ttnn/operations/embedding/embedding.hpp>
#include <ttnn/operations/embedding_backward/embedding_backward.hpp>
#include <ttnn/operations/data_movement/untilize/untilize.hpp>
#include <cmath>
#include <vector>

namespace traced {

using MeshDevice = tt::tt_metal::distributed::MeshDevice;

// Helper to create tensor on device
inline Tensor make_full(ttnn::Shape shape, float value, MeshDevice& device) {
    return ttnn::full(shape, value, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
}

inline Tensor make_zeros(ttnn::Shape shape, MeshDevice& device) {
    return ttnn::zeros(shape, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
}

// Helper to create ROW_MAJOR tensor for embedding weights
// ttnn::embedding requires weight in ROW_MAJOR layout
inline Tensor make_embedding_weight(ttnn::Shape shape, float init_val, MeshDevice& device) {
    // Create in TILE_LAYOUT first, then untilize to ROW_MAJOR
    auto tiled = ttnn::full(shape, init_val, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    return ttnn::untilize(tiled);
}

// Helper to create uint32 index tensor from vector on device
inline Tensor make_indices(const std::vector<uint32_t>& indices, ttnn::Shape shape, MeshDevice& device) {
    auto tensor_layout = ttnn::TensorLayout(
        ttnn::DataType::UINT32,
        ttnn::PageConfig(ttnn::ROW_MAJOR_LAYOUT),
        tt::tt_metal::MemoryConfig{}
    );
    auto tensor = Tensor::from_vector(indices, ttnn::TensorSpec(shape, tensor_layout));
    return tensor.to_device(&device);
}

// TracedLinear: Linear layer with pre-allocated gradient buffers
// Layout: y = x @ weight.T + bias
// weight: [out_features, in_features]
// bias: [1, out_features]
struct TracedLinear {
    // Fields in declaration order (must match initializer order)
    Tensor weight;
    Tensor bias;
    Tensor d_weight;
    Tensor d_bias;
    uint32_t in_features;
    uint32_t out_features;

    TracedLinear(uint32_t in_f, uint32_t out_f, float init, MeshDevice& device)
        : weight(make_full(ttnn::Shape({out_f, in_f}), init, device)),
          bias(make_zeros(ttnn::Shape({1, out_f}), device)),
          d_weight(make_zeros(ttnn::Shape({out_f, in_f}), device)),
          d_bias(make_zeros(ttnn::Shape({1, out_f}), device)),
          in_features(in_f),
          out_features(out_f) {}

    // Forward: out = x @ weight.T + bias
    void forward(const Tensor& x, Tensor& out) {
        out = ttnn::add(ttnn::matmul(x, weight, false, true), bias);
    }

    // Backward: compute gradients and propagate
    void backward(const Tensor& x, const Tensor& d_out, Tensor& d_input) {
        // d_weight = d_out.T @ x
        d_weight = ttnn::matmul(d_out, x, true, false);
        // d_bias = sum(d_out, dim=0)
        d_bias = ttnn::sum(d_out, 0, true);
        // d_input = d_out @ weight
        d_input = ttnn::matmul(d_out, weight);
    }

    // Backward without d_input (for first layer)
    void backward_no_input(const Tensor& x, const Tensor& d_out) {
        d_weight = ttnn::matmul(d_out, x, true, false);
        d_bias = ttnn::sum(d_out, 0, true);
    }

    // SGD update
    void sgd_step(float lr) {
        weight = ttnn::subtract(weight, ttnn::multiply(d_weight, lr));
        bias = ttnn::subtract(bias, ttnn::multiply(d_bias, lr));
    }
};

// TracedCausalAttention: Self-attention with causal mask
// Input shapes: Q, K, V all [batch, heads, seq_len, head_dim]
// Output: [batch, heads, seq_len, head_dim]
//
// Forward:
//   scores = Q @ K.T * scale
//   scores_masked = scores + causal_mask  (upper tri = -inf)
//   attn_weights = softmax(scores_masked, dim=-1)
//   output = attn_weights @ V
//
// Backward:
//   d_attn = d_output @ V.T
//   d_V = attn_weights.T @ d_output
//   d_scores = softmax_backward(d_attn, attn_weights)
//   d_scores_scaled = d_scores * scale
//   d_Q = d_scores_scaled @ K
//   d_K = d_scores_scaled.T @ Q
struct TracedCausalAttention {
    // Forward buffers (declare in init order)
    Tensor scores;           // [B, H, S, S]
    Tensor scores_masked;    // [B, H, S, S]
    Tensor attn_weights;     // [B, H, S, S]
    Tensor output;           // [B, H, S, D]

    // Backward buffers
    Tensor d_attn;           // [B, H, S, S]
    Tensor d_scores;         // [B, H, S, S]
    Tensor d_q;              // [B, H, S, D]
    Tensor d_k;              // [B, H, S, D]
    Tensor d_v;              // [B, H, S, D]

    // Causal mask (static, created once)
    Tensor causal_mask;      // [1, 1, S, S] broadcasts to [B, H, S, S]

    // Config
    float scale;
    uint32_t batch;
    uint32_t heads;
    uint32_t seq_len;
    uint32_t head_dim;

    TracedCausalAttention(uint32_t b, uint32_t h, uint32_t s, uint32_t d, MeshDevice& dev)
        : scores(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          scores_masked(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          attn_weights(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          output(make_zeros(ttnn::Shape({b, h, s, d}), dev)),
          d_attn(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          d_scores(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          d_q(make_zeros(ttnn::Shape({b, h, s, d}), dev)),
          d_k(make_zeros(ttnn::Shape({b, h, s, d}), dev)),
          d_v(make_zeros(ttnn::Shape({b, h, s, d}), dev)),
          causal_mask(create_causal_mask(s, dev)),
          scale(1.0f / std::sqrt(static_cast<float>(d))),
          batch(b),
          heads(h),
          seq_len(s),
          head_dim(d) {}

    // Create upper triangular mask: mask[i,j] = -1e9 if j > i, else 0
    static Tensor create_causal_mask(uint32_t s, MeshDevice& dev) {
        auto mask = make_full(ttnn::Shape({1, 1, s, s}), -1e9f, dev);
        return ttnn::triu(mask, 1);
    }

    // Forward pass
    void forward(const Tensor& q, const Tensor& k, const Tensor& v) {
        // scores = Q @ K.T * scale
        scores = ttnn::multiply(ttnn::matmul(q, k, false, true), scale);
        // Apply causal mask
        scores_masked = ttnn::add(scores, causal_mask);
        // Softmax over last dimension
        traced::softmax(scores_masked, -1, attn_weights);
        // Output = attn_weights @ V
        output = ttnn::matmul(attn_weights, v);
    }

    // Backward pass - computes d_q, d_k, d_v given d_output
    void backward(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& d_out) {
        // d_attn = d_output @ V.T
        d_attn = ttnn::matmul(d_out, v, false, true);
        // d_V = attn_weights.T @ d_output
        d_v = ttnn::matmul(attn_weights, d_out, true, false);

        // Softmax backward
        traced::softmax_backward(d_attn, attn_weights, -1, d_scores);

        // Scale gradients
        auto d_scores_scaled = ttnn::multiply(d_scores, scale);

        // d_Q = d_scores_scaled @ K
        d_q = ttnn::matmul(d_scores_scaled, k);
        // d_K = d_scores_scaled.T @ Q
        d_k = ttnn::matmul(d_scores_scaled, q, true, false);
    }
};

// TracedLayerNorm: Layer normalization with learnable gamma/beta
// Manual implementation using basic TTNN ops for reliability.
// Input: [B, S, D]
// Normalizes over the last dimension
//
// Forward: out = gamma * (x - mean) / sqrt(var + eps) + beta
// Backward:
//   d_gamma = sum(d_out * x_norm, dims=[0,1])
//   d_beta = sum(d_out, dims=[0,1])
//   d_x = (1/sqrt(var+eps)) * (d_out * gamma - mean(d_out * gamma) - x_norm * mean(d_out * gamma * x_norm))
struct TracedLayerNorm {
    Tensor gamma;       // [1, 1, D] - learnable scale
    Tensor beta;        // [1, 1, D] - learnable bias
    Tensor d_gamma;     // gradient for gamma
    Tensor d_beta;      // gradient for beta
    Tensor x_norm;      // [B, S, D] - cached normalized input for backward
    Tensor rstd;        // [B, S, 1] - cached 1/sqrt(var+eps) for backward
    uint32_t dim;
    float eps;

    TracedLayerNorm(uint32_t d, float epsilon, MeshDevice& dev)
        : gamma(make_full(ttnn::Shape({1, 1, d}), 1.0f, dev)),
          beta(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          d_gamma(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          d_beta(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          x_norm(make_zeros(ttnn::Shape({1, 1, d}), dev)),  // Will be set in forward
          rstd(make_zeros(ttnn::Shape({1, 1, 1}), dev)),    // Will be set in forward
          dim(d),
          eps(epsilon) {}

    // Forward: out = gamma * (x - mean) / sqrt(var + eps) + beta
    void forward(const Tensor& x, Tensor& out) {
        // mean = mean(x, dim=-1, keepdim=True)
        auto mean = ttnn::mean(x, -1, true);

        // x_centered = x - mean
        auto x_centered = ttnn::subtract(x, mean);

        // var = mean(x_centered^2, dim=-1, keepdim=True)
        auto var = ttnn::mean(ttnn::multiply(x_centered, x_centered), -1, true);

        // rstd = 1 / sqrt(var + eps)
        rstd = ttnn::rsqrt(ttnn::add(var, eps));

        // x_norm = x_centered * rstd
        x_norm = ttnn::multiply(x_centered, rstd);

        // out = gamma * x_norm + beta
        out = ttnn::add(ttnn::multiply(gamma, x_norm), beta);
    }

    // Backward: compute d_input, d_gamma, d_beta
    void backward(const Tensor& x, const Tensor& d_out, Tensor& d_input) {
        // d_beta = sum(d_out, dims=[0,1], keepdim=True)
        d_beta = ttnn::sum(ttnn::sum(d_out, 0, true), 1, true);

        // d_gamma = sum(d_out * x_norm, dims=[0,1], keepdim=True)
        d_gamma = ttnn::sum(ttnn::sum(ttnn::multiply(d_out, x_norm), 0, true), 1, true);

        // d_x_norm = d_out * gamma
        auto d_x_norm = ttnn::multiply(d_out, gamma);

        // LayerNorm backward formula (simplified):
        // d_x = rstd * (d_x_norm - mean(d_x_norm) - x_norm * mean(d_x_norm * x_norm))
        auto mean_d_x_norm = ttnn::mean(d_x_norm, -1, true);
        auto mean_d_x_norm_x_norm = ttnn::mean(ttnn::multiply(d_x_norm, x_norm), -1, true);

        d_input = ttnn::multiply(
            rstd,
            ttnn::subtract(
                ttnn::subtract(d_x_norm, mean_d_x_norm),
                ttnn::multiply(x_norm, mean_d_x_norm_x_norm)
            )
        );
    }

    // SGD update
    void sgd_step(float lr) {
        gamma = ttnn::subtract(gamma, ttnn::multiply(d_gamma, lr));
        beta = ttnn::subtract(beta, ttnn::multiply(d_beta, lr));
    }
};

// TracedEmbedding: Embedding layer with trainable weights
// Input: uint32 indices [batch, seq]
// Output: bfloat16 [batch, seq, embedding_dim]
// Weight: [vocab_size, embedding_dim] in ROW_MAJOR layout
//
// Note: embedding_bw requires gradient shape [1, 1, batch*seq, dim]
struct TracedEmbedding {
    Tensor weight;         // [vocab_size, embedding_dim] ROW_MAJOR
    Tensor d_weight;       // gradient buffer [vocab_size, embedding_dim]
    uint32_t vocab_size;
    uint32_t embedding_dim;
    uint32_t batch_size;
    uint32_t seq_len;

    TracedEmbedding(uint32_t vocab, uint32_t dim, uint32_t b, uint32_t s, float init_val, MeshDevice& dev)
        : weight(make_embedding_weight(ttnn::Shape({vocab, dim}), init_val, dev)),
          d_weight(make_zeros(ttnn::Shape({vocab, dim}), dev)),
          vocab_size(vocab),
          embedding_dim(dim),
          batch_size(b),
          seq_len(s) {}

    // Forward: lookup embeddings for input indices
    // indices: uint32 [batch, seq] in ROW_MAJOR
    // out: bfloat16 [batch, seq, embedding_dim] in TILE_LAYOUT
    void forward(const Tensor& indices, Tensor& out) {
        out = ttnn::embedding(indices, weight, std::nullopt, ttnn::TILE_LAYOUT);
    }

    // Backward: compute weight gradients (indices have no gradient)
    // indices: uint32 [batch, seq]
    // d_out: bfloat16 [batch, seq, embedding_dim]
    void backward(const Tensor& indices, const Tensor& d_out) {
        // embedding_bw requires gradient shape [1, 1, batch*seq, dim]
        auto d_out_reshaped = ttnn::reshape(d_out, ttnn::Shape({1, 1, batch_size * seq_len, embedding_dim}));
        d_weight = ttnn::embedding_bw(indices, weight, d_out_reshaped, ttnn::DataType::BFLOAT16);
    }

    // SGD update - need to convert d_weight from TILE to ROW_MAJOR for update
    void sgd_step(float lr) {
        // d_weight is in TILE_LAYOUT, need to untilize for update
        auto d_weight_rm = ttnn::untilize(d_weight);
        weight = ttnn::subtract(weight, ttnn::multiply(d_weight_rm, lr));
    }
};

}  // namespace traced
