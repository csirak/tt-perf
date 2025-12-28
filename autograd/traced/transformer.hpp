// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// TracedTransformerLayer: Forward-only transformer layer for trace region benchmarking
// Pre-allocates all buffers for trace compatibility.
//
// Architecture:
//   input -> Attention -> Add(residual) -> FFN -> Add(residual) -> output
//
// No LayerNorm - using identity for simplicity.

#pragma once

#include "nn.hpp"  // For make_full, make_zeros, softmax
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>  // For ttnn::triu
#include <cmath>

namespace traced {

using MeshDevice = tt::tt_metal::distributed::MeshDevice;

// =============================================================================
// DynamicTransformerLayer: Allocates new tensors each forward pass
// Used as baseline to show speedup from pre-allocation + trace.
// =============================================================================
struct DynamicTransformerLayer {
    // Only weights - no intermediate buffers (allocated each forward)
    Tensor wq, wk, wv, wo;
    Tensor w1, w2;
    Tensor causal_mask;

    float scale;
    uint32_t batch, seq, dim, heads, head_dim, ffn_dim;

    DynamicTransformerLayer(uint32_t b, uint32_t s, uint32_t d, uint32_t h,
                            uint32_t ffn_mult, float init_val, MeshDevice& dev)
        : scale(1.0f / std::sqrt(static_cast<float>(d / h))),
          batch(b), seq(s), dim(d), heads(h), head_dim(d / h), ffn_dim(d * ffn_mult)
    {
        wq = make_full(ttnn::Shape({d, d}), init_val, dev);
        wk = make_full(ttnn::Shape({d, d}), init_val, dev);
        wv = make_full(ttnn::Shape({d, d}), init_val, dev);
        wo = make_full(ttnn::Shape({d, d}), init_val, dev);
        w1 = make_full(ttnn::Shape({d, ffn_dim}), init_val, dev);
        w2 = make_full(ttnn::Shape({ffn_dim, d}), init_val, dev);
        causal_mask = create_causal_mask(s, dev);
    }

    static Tensor create_causal_mask(uint32_t s, MeshDevice& dev) {
        auto mask = make_full(ttnn::Shape({1, 1, s, s}), -1e9f, dev);
        return ttnn::triu(mask, 1);
    }

    // Forward pass - allocates new tensors for all intermediates
    Tensor forward(const Tensor& input) {
        // Q, K, V projections
        auto q_proj = ttnn::matmul(input, wq);
        auto k_proj = ttnn::matmul(input, wk);
        auto v_proj = ttnn::matmul(input, wv);

        // Reshape to [B, H, S, D/H]
        auto q = ttnn::transpose(
            ttnn::reshape(q_proj, ttnn::Shape({batch, seq, heads, head_dim})), 1, 2);
        auto k = ttnn::transpose(
            ttnn::reshape(k_proj, ttnn::Shape({batch, seq, heads, head_dim})), 1, 2);
        auto v = ttnn::transpose(
            ttnn::reshape(v_proj, ttnn::Shape({batch, seq, heads, head_dim})), 1, 2);

        // Attention scores
        auto scores = ttnn::multiply(ttnn::matmul(q, k, false, true), scale);
        auto scores_masked = ttnn::add(scores, causal_mask);

        // Softmax (with max-centering for stability)
        auto x_max = ttnn::max(scores_masked, -1, true);
        auto x_centered = ttnn::subtract(scores_masked, x_max);
        auto attn_weights = ttnn::softmax(x_centered, -1);

        // Attention output
        auto attn_out = ttnn::matmul(attn_weights, v);

        // Merge heads
        auto attn_merged = ttnn::reshape(
            ttnn::transpose(attn_out, 1, 2),
            ttnn::Shape({batch, seq, dim}));
        auto attn_proj = ttnn::matmul(attn_merged, wo);

        // Residual 1
        auto residual1 = ttnn::add(input, attn_proj);

        // FFN
        auto h1 = ttnn::matmul(residual1, w1);
        auto h1_relu = ttnn::relu(h1);
        auto ffn_out = ttnn::matmul(h1_relu, w2);

        // Residual 2
        return ttnn::add(residual1, ffn_out);
    }
};

// =============================================================================
// TracedTransformerLayer: Single transformer layer with pre-allocated buffers
// Shapes: input [B, S, D], output [B, S, D]
// Attention: Q, K, V projections, scaled dot-product with causal mask
// FFN: two linear layers with ReLU activation
// =============================================================================
struct TracedTransformerLayer {
    // Attention weights [D, D] - stored as [out, in] for matmul(x, w, false, true)
    Tensor wq, wk, wv, wo;

    // FFN weights
    Tensor w1;  // [D, D*ffn_mult]
    Tensor w2;  // [D*ffn_mult, D]

    // Attention intermediate buffers
    Tensor q_proj;        // [B, S, D] after projection
    Tensor k_proj;        // [B, S, D]
    Tensor v_proj;        // [B, S, D]
    Tensor q;             // [B, H, S, D/H] after reshape
    Tensor k;             // [B, H, S, D/H]
    Tensor v;             // [B, H, S, D/H]
    Tensor scores;        // [B, H, S, S]
    Tensor scores_masked; // [B, H, S, S]
    Tensor attn_weights;  // [B, H, S, S]
    Tensor attn_out;      // [B, H, S, D/H]
    Tensor attn_merged;   // [B, S, D] after merge heads
    Tensor attn_proj;     // [B, S, D] after output projection

    // FFN intermediate buffers
    Tensor h1;            // [B, S, D*ffn_mult]
    Tensor h1_relu;       // [B, S, D*ffn_mult]
    Tensor ffn_out;       // [B, S, D]

    // Residual outputs
    Tensor residual1;     // [B, S, D] after attention residual
    Tensor output;        // [B, S, D] final output

    // Causal mask (static, created once)
    Tensor causal_mask;   // [1, 1, S, S] broadcasts to [B, H, S, S]

    // Config
    float scale;
    uint32_t batch, seq, dim, heads, head_dim, ffn_dim;

    TracedTransformerLayer(uint32_t b, uint32_t s, uint32_t d, uint32_t h,
                           uint32_t ffn_mult, float init_val, MeshDevice& dev)
        : scale(1.0f / std::sqrt(static_cast<float>(d / h))),
          batch(b), seq(s), dim(d), heads(h), head_dim(d / h), ffn_dim(d * ffn_mult)
    {
        // Attention weights
        wq = make_full(ttnn::Shape({d, d}), init_val, dev);
        wk = make_full(ttnn::Shape({d, d}), init_val, dev);
        wv = make_full(ttnn::Shape({d, d}), init_val, dev);
        wo = make_full(ttnn::Shape({d, d}), init_val, dev);

        // FFN weights
        w1 = make_full(ttnn::Shape({d, ffn_dim}), init_val, dev);
        w2 = make_full(ttnn::Shape({ffn_dim, d}), init_val, dev);

        // Attention buffers
        q_proj = make_zeros(ttnn::Shape({b, s, d}), dev);
        k_proj = make_zeros(ttnn::Shape({b, s, d}), dev);
        v_proj = make_zeros(ttnn::Shape({b, s, d}), dev);
        q = make_zeros(ttnn::Shape({b, h, s, head_dim}), dev);
        k = make_zeros(ttnn::Shape({b, h, s, head_dim}), dev);
        v = make_zeros(ttnn::Shape({b, h, s, head_dim}), dev);
        scores = make_zeros(ttnn::Shape({b, h, s, s}), dev);
        scores_masked = make_zeros(ttnn::Shape({b, h, s, s}), dev);
        attn_weights = make_zeros(ttnn::Shape({b, h, s, s}), dev);
        attn_out = make_zeros(ttnn::Shape({b, h, s, head_dim}), dev);
        attn_merged = make_zeros(ttnn::Shape({b, s, d}), dev);
        attn_proj = make_zeros(ttnn::Shape({b, s, d}), dev);

        // FFN buffers
        h1 = make_zeros(ttnn::Shape({b, s, ffn_dim}), dev);
        h1_relu = make_zeros(ttnn::Shape({b, s, ffn_dim}), dev);
        ffn_out = make_zeros(ttnn::Shape({b, s, d}), dev);

        // Residual outputs
        residual1 = make_zeros(ttnn::Shape({b, s, d}), dev);
        output = make_zeros(ttnn::Shape({b, s, d}), dev);

        // Causal mask: upper triangular with -1e9
        causal_mask = create_causal_mask(s, dev);
    }

    // Create causal mask: [1, 1, S, S] with -1e9 above diagonal
    static Tensor create_causal_mask(uint32_t s, MeshDevice& dev) {
        auto mask = make_full(ttnn::Shape({1, 1, s, s}), -1e9f, dev);
        return ttnn::triu(mask, 1);
    }

    // Forward pass: input [B, S, D] -> output [B, S, D]
    void forward(const Tensor& input) {
        // Q, K, V projections: [B, S, D] @ [D, D] -> [B, S, D]
        q_proj = ttnn::matmul(input, wq);
        k_proj = ttnn::matmul(input, wk);
        v_proj = ttnn::matmul(input, wv);

        // Reshape to [B, H, S, D/H] for multi-head attention
        // [B, S, D] -> [B, S, H, D/H] -> [B, H, S, D/H]
        q = ttnn::transpose(
            ttnn::reshape(q_proj, ttnn::Shape({batch, seq, heads, head_dim})),
            1, 2);
        k = ttnn::transpose(
            ttnn::reshape(k_proj, ttnn::Shape({batch, seq, heads, head_dim})),
            1, 2);
        v = ttnn::transpose(
            ttnn::reshape(v_proj, ttnn::Shape({batch, seq, heads, head_dim})),
            1, 2);

        // Attention scores: Q @ K.T * scale -> [B, H, S, S]
        scores = ttnn::multiply(ttnn::matmul(q, k, false, true), scale);

        // Apply causal mask
        scores_masked = ttnn::add(scores, causal_mask);

        // Softmax over last dimension
        softmax(scores_masked, -1, attn_weights);

        // Attention output: attn @ V -> [B, H, S, D/H]
        attn_out = ttnn::matmul(attn_weights, v);

        // Merge heads: [B, H, S, D/H] -> [B, S, H, D/H] -> [B, S, D]
        attn_merged = ttnn::reshape(
            ttnn::transpose(attn_out, 1, 2),
            ttnn::Shape({batch, seq, dim}));

        // Output projection: [B, S, D] @ [D, D] -> [B, S, D]
        attn_proj = ttnn::matmul(attn_merged, wo);

        // Residual 1
        residual1 = ttnn::add(input, attn_proj);

        // FFN: Linear -> ReLU -> Linear
        h1 = ttnn::matmul(residual1, w1);
        h1_relu = ttnn::relu(h1);
        ffn_out = ttnn::matmul(h1_relu, w2);

        // Residual 2
        output = ttnn::add(residual1, ffn_out);
    }

    // Get output tensor (for chaining layers)
    const Tensor& get_output() const { return output; }

    // Read output values (syncs device)
    float get_first_output(MeshDevice* dev) {
        tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        return static_cast<float>(output.cpu().to_vector<bfloat16>()[0]);
    }
};

}  // namespace traced
