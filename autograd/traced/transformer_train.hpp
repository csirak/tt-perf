// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// TracedTransformerLayerTrainable: Transformer layer with forward/backward/SGD
// For benchmarking full training step vs TTML nano_gpt.
//
// Architecture (GPT-2 style with pre-norm):
//   input -> LN1 -> Attention -> Add(residual) -> LN2 -> FFN(GELU) -> Add(residual) -> output
//
// Shapes:
//   input:  [B, S, D]
//   output: [B, S, D]

#pragma once

#include "nn.hpp"
#include "ops.hpp"
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <cmath>
#include <vector>

namespace traced {

// Shared attention buffers (used across all layers to save memory)
// Created once, passed to each layer during forward/backward
struct SharedAttentionBuffers {
    Tensor scores;         // [B, H, S, S]
    Tensor scores_masked;  // [B, H, S, S]
    Tensor attn_weights;   // [B, H, S, S]
    Tensor causal_mask;    // [1, 1, S, S]

    // Backward buffers
    Tensor d_attn;         // [B, H, S, S]
    Tensor d_scores;       // [B, H, S, S]

    float scale;

    SharedAttentionBuffers(uint32_t b, uint32_t h, uint32_t s, uint32_t d, MeshDevice& dev)
        : scores(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          scores_masked(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          attn_weights(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          causal_mask(create_causal_mask(s, dev)),
          d_attn(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          d_scores(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          scale(1.0f / std::sqrt(static_cast<float>(d))) {}

    static Tensor create_causal_mask(uint32_t s, MeshDevice& dev) {
        auto mask = make_full(ttnn::Shape({1, 1, s, s}), -1e9f, dev);
        return ttnn::triu(mask, 1);
    }

    // Attention forward using shared buffers
    void attention_forward(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& output) {
        scores = ttnn::multiply(ttnn::matmul(q, k, false, true), scale);
        scores_masked = ttnn::add(scores, causal_mask);
        traced::softmax(scores_masked, -1, attn_weights);
        output = ttnn::matmul(attn_weights, v);
    }

    // Attention backward using shared buffers
    void attention_backward(const Tensor& q, const Tensor& k, const Tensor& v,
                           const Tensor& d_out, Tensor& d_q, Tensor& d_k, Tensor& d_v) {
        d_attn = ttnn::matmul(d_out, v, false, true);
        d_v = ttnn::matmul(attn_weights, d_out, true, false);
        traced::softmax_backward(d_attn, attn_weights, -1, d_scores);
        auto d_scores_scaled = ttnn::multiply(d_scores, scale);
        d_q = ttnn::matmul(d_scores_scaled, k);
        d_k = ttnn::matmul(d_scores_scaled, q, true, false);
    }
};

// Single transformer layer with backward pass support
// Uses shared attention buffers to reduce memory
// Architecture: LN1 -> Attention -> Add -> LN2 -> FFN(GELU) -> Add
struct TracedTransformerLayerTrainable {
    // LayerNorm modules
    TracedLayerNorm ln1, ln2;

    // Attention linear layers
    TracedLinear wq, wk, wv, wo;

    // FFN linear layers
    TracedLinear w1, w2;

    // Forward buffers (layer-specific - needed for backward)
    Tensor ln1_out;                       // [B, S, D] LN1 output
    Tensor q_proj, k_proj, v_proj;        // [B, S, D] QKV projections
    Tensor q, k, v;                       // [B, H, S, D/H] reshaped for attention
    Tensor attn_out;                      // [B, H, S, D/H] attention output
    Tensor attn_merged;                   // [B, S, D] merged heads
    Tensor attn_proj;                     // [B, S, D] output projection
    Tensor residual1;                     // [B, S, D] after first residual
    Tensor ln2_out;                       // [B, S, D] LN2 output
    Tensor h1, h1_gelu;                   // [B, S, D*4] FFN hidden with GELU
    Tensor ffn_out;                       // [B, S, D] FFN output
    Tensor output;                        // [B, S, D] final output

    // Backward buffers
    Tensor d_ffn_out;                     // [B, S, D]
    Tensor d_h1_gelu, d_h1;               // [B, S, D*4]
    Tensor d_ln2_out;                     // [B, S, D]
    Tensor d_residual1;                   // [B, S, D]
    Tensor d_attn_proj;                   // [B, S, D]
    Tensor d_attn_merged;                 // [B, S, D]
    Tensor d_attn_out;                    // [B, H, S, D/H]
    Tensor d_q, d_k, d_v;                 // [B, H, S, D/H] attention grads
    Tensor d_q_proj, d_k_proj, d_v_proj;  // [B, S, D]
    Tensor d_x_q, d_x_k, d_x_v;           // [B, S, D] gradients w.r.t. LN1 output
    Tensor d_ln1_out;                     // [B, S, D] gradient for LN1 backward
    Tensor d_input;                       // [B, S, D] final input gradient

    // Config
    uint32_t batch, seq, dim, heads, head_dim, ffn_dim;
    float lr;

    TracedTransformerLayerTrainable(uint32_t b, uint32_t s, uint32_t d, uint32_t h,
                                    uint32_t ffn_mult, float learning_rate, MeshDevice& dev)
        : ln1(d, 1e-5f, dev),
          ln2(d, 1e-5f, dev),
          wq(d, d, 0.01f, dev),
          wk(d, d, 0.01f, dev),
          wv(d, d, 0.01f, dev),
          wo(d, d, 0.01f, dev),
          w1(d, d * ffn_mult, 0.01f, dev),
          w2(d * ffn_mult, d, 0.01f, dev),
          // Forward buffers
          ln1_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          q_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          k_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          v_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          q(make_zeros(ttnn::Shape({b, h, s, d / h}), dev)),
          k(make_zeros(ttnn::Shape({b, h, s, d / h}), dev)),
          v(make_zeros(ttnn::Shape({b, h, s, d / h}), dev)),
          attn_out(make_zeros(ttnn::Shape({b, h, s, d / h}), dev)),
          attn_merged(make_zeros(ttnn::Shape({b, s, d}), dev)),
          attn_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          residual1(make_zeros(ttnn::Shape({b, s, d}), dev)),
          ln2_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          h1(make_zeros(ttnn::Shape({b, s, d * ffn_mult}), dev)),
          h1_gelu(make_zeros(ttnn::Shape({b, s, d * ffn_mult}), dev)),
          ffn_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          output(make_zeros(ttnn::Shape({b, s, d}), dev)),
          // Backward buffers
          d_ffn_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_h1_gelu(make_zeros(ttnn::Shape({b, s, d * ffn_mult}), dev)),
          d_h1(make_zeros(ttnn::Shape({b, s, d * ffn_mult}), dev)),
          d_ln2_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_residual1(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_attn_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_attn_merged(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_attn_out(make_zeros(ttnn::Shape({b, h, s, d / h}), dev)),
          d_q(make_zeros(ttnn::Shape({b, h, s, d / h}), dev)),
          d_k(make_zeros(ttnn::Shape({b, h, s, d / h}), dev)),
          d_v(make_zeros(ttnn::Shape({b, h, s, d / h}), dev)),
          d_q_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_k_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_v_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_x_q(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_x_k(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_x_v(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_ln1_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_input(make_zeros(ttnn::Shape({b, s, d}), dev)),
          batch(b), seq(s), dim(d), heads(h), head_dim(d / h), ffn_dim(d * ffn_mult),
          lr(learning_rate) {}

    // Reshape [B, S, D] -> [B, H, S, D/H]
    Tensor reshape_to_heads(const Tensor& x) {
        auto reshaped = ttnn::reshape(x, ttnn::Shape({batch, seq, heads, head_dim}));
        return ttnn::transpose(reshaped, 1, 2);  // [B, S, H, D/H] -> [B, H, S, D/H]
    }

    // Reshape [B, H, S, D/H] -> [B, S, D]
    Tensor merge_heads(const Tensor& x) {
        auto transposed = ttnn::transpose(x, 1, 2);  // [B, H, S, D/H] -> [B, S, H, D/H]
        return ttnn::reshape(transposed, ttnn::Shape({batch, seq, dim}));
    }

    void forward(const Tensor& x, SharedAttentionBuffers& attn_buf) {
        // Pre-attention LayerNorm
        ln1.forward(x, ln1_out);

        // QKV projections on normalized input
        wq.forward(ln1_out, q_proj);
        wk.forward(ln1_out, k_proj);
        wv.forward(ln1_out, v_proj);

        // Reshape to multi-head format
        q = reshape_to_heads(q_proj);
        k = reshape_to_heads(k_proj);
        v = reshape_to_heads(v_proj);

        // Causal attention using shared buffers
        attn_buf.attention_forward(q, k, v, attn_out);

        // Merge heads and project
        attn_merged = merge_heads(attn_out);
        wo.forward(attn_merged, attn_proj);

        // Residual 1
        residual1 = ttnn::add(x, attn_proj);

        // Pre-FFN LayerNorm
        ln2.forward(residual1, ln2_out);

        // FFN: Linear -> GELU -> Linear
        w1.forward(ln2_out, h1);
        traced::gelu(h1, h1_gelu);
        w2.forward(h1_gelu, ffn_out);

        // Residual 2
        output = ttnn::add(residual1, ffn_out);
    }

    void backward(const Tensor& x, const Tensor& d_out, SharedAttentionBuffers& attn_buf) {
        // Residual 2 backward: gradient flows to both ffn_out and residual1
        d_ffn_out = d_out;

        // FFN backward with GELU
        w2.backward(h1_gelu, d_ffn_out, d_h1_gelu);
        traced::gelu_backward(d_h1_gelu, h1, d_h1);
        w1.backward(ln2_out, d_h1, d_ln2_out);

        // Pre-FFN LayerNorm backward
        ln2.backward(residual1, d_ln2_out, d_residual1);

        // Add gradient from residual connection
        d_residual1 = ttnn::add(d_residual1, d_out);

        // Attention projection backward
        wo.backward(attn_merged, d_residual1, d_attn_merged);

        // Reshape gradient to multi-head format
        d_attn_out = reshape_to_heads(d_attn_merged);

        // Attention backward using shared buffers
        attn_buf.attention_backward(q, k, v, d_attn_out, d_q, d_k, d_v);

        // Reshape gradients back to [B, S, D]
        d_q_proj = merge_heads(d_q);
        d_k_proj = merge_heads(d_k);
        d_v_proj = merge_heads(d_v);

        // QKV projection backward (input was ln1_out, not x)
        wq.backward(ln1_out, d_q_proj, d_x_q);
        wk.backward(ln1_out, d_k_proj, d_x_k);
        wv.backward(ln1_out, d_v_proj, d_x_v);

        // Sum gradients to LN1 output
        d_ln1_out = ttnn::add(ttnn::add(d_x_q, d_x_k), d_x_v);

        // Pre-attention LayerNorm backward
        ln1.backward(x, d_ln1_out, d_input);

        // Add residual gradient (from both residual connections)
        d_input = ttnn::add(d_input, d_residual1);
    }

    void sgd_step() {
        ln1.sgd_step(lr);
        ln2.sgd_step(lr);
        wq.sgd_step(lr);
        wk.sgd_step(lr);
        wv.sgd_step(lr);
        wo.sgd_step(lr);
        w1.sgd_step(lr);
        w2.sgd_step(lr);
    }

    const Tensor& get_output() const { return output; }
    const Tensor& get_d_input() const { return d_input; }
};

// Stack of N transformer layers with training support
template<size_t N>
struct TracedTransformerStack {
    static_assert(N >= 1, "Must have at least 1 layer");

    // Shared attention buffers (one set for all layers)
    SharedAttentionBuffers attn_buf;

    std::vector<TracedTransformerLayerTrainable> layers;

    // Loss computation buffers
    Tensor diff;
    Tensor sq;
    Tensor loss;
    Tensor d_loss;  // gradient of loss w.r.t. output

    // Config
    float loss_scale;
    uint32_t batch, seq, dim;

    TracedTransformerStack(uint32_t b, uint32_t s, uint32_t d, uint32_t h,
                           uint32_t ffn_mult, float lr, MeshDevice& dev)
        : attn_buf(b, h, s, d / h, dev),
          diff(make_zeros(ttnn::Shape({b, s, d}), dev)),
          sq(make_zeros(ttnn::Shape({b, s, d}), dev)),
          loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          d_loss(make_zeros(ttnn::Shape({b, s, d}), dev)),
          loss_scale(1.0f / (b * s * d)),
          batch(b), seq(s), dim(d)
    {
        layers.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            layers.emplace_back(b, s, d, h, ffn_mult, lr, dev);
        }
    }

    void forward(const Tensor& x) {
        layers[0].forward(x, attn_buf);
        for (size_t i = 1; i < N; ++i) {
            layers[i].forward(layers[i - 1].get_output(), attn_buf);
        }
    }

    void compute_loss(const Tensor& target) {
        traced::subtract(layers[N - 1].get_output(), target, diff);
        traced::multiply(diff, diff, sq);
        traced::mean(sq, loss);
    }

    void backward(const Tensor& x) {
        // MSE gradient: d_output = 2 * diff * loss_scale
        traced::mse_backward(diff, loss_scale, d_loss);

        // Backprop through layers in reverse
        layers[N - 1].backward(N > 1 ? layers[N - 2].get_output() : x, d_loss, attn_buf);
        for (size_t i = N - 1; i > 0; --i) {
            const Tensor& layer_input = (i > 1) ? layers[i - 2].get_output() : x;
            layers[i - 1].backward(layer_input, layers[i].get_d_input(), attn_buf);
        }
    }

    void sgd_step() {
        for (size_t i = 0; i < N; ++i) {
            layers[i].sgd_step();
        }
    }

    void train_step(const Tensor& x, const Tensor& target) {
        forward(x);
        compute_loss(target);
        backward(x);
        sgd_step();
    }

    float get_loss(MeshDevice* dev) {
        tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        return static_cast<float>(loss.cpu().to_vector<bfloat16>()[0]);
    }
};

// Full GPT-2 style model with embeddings for fair TTML comparison
// Architecture:
//   token_ids -> tok_emb + pos_emb -> N x TransformerLayer -> output_proj -> logits
//
// Matches TTML nano_gpt:
//   - Token embedding [vocab_size, dim]
//   - Positional embedding [max_seq, dim]
//   - N transformer layers (LN + Attention + LN + FFN)
//   - Output projection [dim, vocab_size]
template<size_t N>
struct TracedTransformerWithEmbedding {
    static_assert(N >= 1, "Must have at least 1 layer");

    // Embeddings
    TracedEmbedding tok_emb;     // [vocab_size, dim]
    TracedEmbedding pos_emb;     // [max_seq, dim]

    // Transformer stack
    SharedAttentionBuffers attn_buf;
    std::vector<TracedTransformerLayerTrainable> layers;

    // Output projection (no bias to match TTML)
    TracedLinear output_proj;    // [dim, vocab_size]

    // Forward buffers
    Tensor tok_emb_out;          // [B, S, D]
    Tensor pos_emb_out;          // [B, S, D]
    Tensor combined_emb;         // [B, S, D]
    Tensor logits;               // [B, S, vocab_size]

    // Positional indices - created once, reused
    Tensor pos_indices;          // [B, S] uint32

    // Loss computation buffers
    Tensor diff;
    Tensor sq;
    Tensor loss;
    Tensor d_loss;               // [B, S, vocab_size]
    Tensor d_combined_emb;       // [B, S, D]

    // Config
    uint32_t batch, seq, dim, vocab_size;
    float loss_scale;
    float lr;

    TracedTransformerWithEmbedding(uint32_t b, uint32_t s, uint32_t d, uint32_t h,
                                    uint32_t ffn_mult, uint32_t vocab, float learning_rate,
                                    MeshDevice& dev)
        : tok_emb(vocab, d, b, s, 0.02f, dev),
          pos_emb(s, d, b, s, 0.02f, dev),  // max_seq = s
          attn_buf(b, h, s, d / h, dev),
          output_proj(d, vocab, 0.02f, dev),
          tok_emb_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          pos_emb_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          combined_emb(make_zeros(ttnn::Shape({b, s, d}), dev)),
          logits(make_zeros(ttnn::Shape({b, s, vocab}), dev)),
          pos_indices(create_position_indices(b, s, dev)),
          diff(make_zeros(ttnn::Shape({b, s, vocab}), dev)),
          sq(make_zeros(ttnn::Shape({b, s, vocab}), dev)),
          loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          d_loss(make_zeros(ttnn::Shape({b, s, vocab}), dev)),
          d_combined_emb(make_zeros(ttnn::Shape({b, s, d}), dev)),
          batch(b), seq(s), dim(d), vocab_size(vocab),
          loss_scale(1.0f / (b * s * vocab)),
          lr(learning_rate)
    {
        layers.reserve(N);
        for (size_t i = 0; i < N; ++i) {
            layers.emplace_back(b, s, d, h, ffn_mult, learning_rate, dev);
        }
    }

    // Create position indices [B, S] with values 0, 1, 2, ..., S-1 repeated for each batch
    static Tensor create_position_indices(uint32_t b, uint32_t s, MeshDevice& dev) {
        std::vector<uint32_t> indices(b * s);
        for (uint32_t i = 0; i < b; ++i) {
            for (uint32_t j = 0; j < s; ++j) {
                indices[i * s + j] = j;
            }
        }
        return make_indices(indices, ttnn::Shape({b, s}), dev);
    }

    void forward(const Tensor& token_ids) {
        // Token embedding
        tok_emb.forward(token_ids, tok_emb_out);

        // Positional embedding
        pos_emb.forward(pos_indices, pos_emb_out);

        // Combine: tok_emb + pos_emb
        combined_emb = ttnn::add(tok_emb_out, pos_emb_out);

        // Transformer layers
        layers[0].forward(combined_emb, attn_buf);
        for (size_t i = 1; i < N; ++i) {
            layers[i].forward(layers[i - 1].get_output(), attn_buf);
        }

        // Output projection to vocab
        output_proj.forward(layers[N - 1].get_output(), logits);
    }

    void compute_loss(const Tensor& target_logits) {
        traced::subtract(logits, target_logits, diff);
        traced::multiply(diff, diff, sq);
        traced::mean(sq, loss);
    }

    void backward(const Tensor& token_ids) {
        // MSE gradient
        traced::mse_backward(diff, loss_scale, d_loss);

        // Output projection backward
        Tensor d_transformer_out;
        output_proj.backward(layers[N - 1].get_output(), d_loss, d_transformer_out);

        // Transformer layers backward
        layers[N - 1].backward(N > 1 ? layers[N - 2].get_output() : combined_emb,
                               d_transformer_out, attn_buf);
        for (size_t i = N - 1; i > 0; --i) {
            const Tensor& layer_input = (i > 1) ? layers[i - 2].get_output() : combined_emb;
            layers[i - 1].backward(layer_input, layers[i].get_d_input(), attn_buf);
        }

        // Get gradient for combined embedding
        d_combined_emb = layers[0].get_d_input();

        // Embedding backward (gradient flows to both tok_emb and pos_emb)
        tok_emb.backward(token_ids, d_combined_emb);
        pos_emb.backward(pos_indices, d_combined_emb);
    }

    void sgd_step() {
        tok_emb.sgd_step(lr);
        pos_emb.sgd_step(lr);
        for (size_t i = 0; i < N; ++i) {
            layers[i].sgd_step();
        }
        output_proj.sgd_step(lr);
    }

    void train_step(const Tensor& token_ids, const Tensor& target_logits) {
        forward(token_ids);
        compute_loss(target_logits);
        backward(token_ids);
        sgd_step();
    }

    float get_loss(MeshDevice* dev) {
        tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        return static_cast<float>(loss.cpu().to_vector<bfloat16>()[0]);
    }
};

}  // namespace traced
