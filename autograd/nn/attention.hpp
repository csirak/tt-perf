#pragma once

#include "common.hpp"

namespace static_autograd {

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

    Tensor* execute_forward(const Tensor& x) {
        auto mean1 = ttnn::mean(x, -1, true);
        auto centered1 = ttnn::subtract(x, mean1);
        auto var1 = ttnn::mean(ttnn::multiply(centered1, centered1), -1, true);
        ln1.rstd = ttnn::rsqrt(ttnn::add(var1, ln1.eps), true);
        ln1.x_norm = ttnn::multiply(centered1, ln1.rstd);
        ln1.out = ttnn::add(ttnn::multiply(ln1.gamma, ln1.x_norm), ln1.beta);

        wq.out = ttnn::add(ttnn::matmul(ln1.out, wq.weight, false, true), wq.bias);
        wk.out = ttnn::add(ttnn::matmul(ln1.out, wk.weight, false, true), wk.bias);
        wv.out = ttnn::add(ttnn::matmul(ln1.out, wv.weight, false, true), wv.bias);

        attn_scores = ttnn::multiply(ttnn::matmul(wq.out, wk.out, false, true), scale);
        auto scores_masked = ttnn::add(attn_scores, causal_mask);
        auto scores_max = ttnn::max(scores_masked, -1, true);
        auto scores_centered = ttnn::subtract(scores_masked, scores_max);
        attn_weights = ttnn::softmax(scores_centered, -1);

        attn_out = ttnn::matmul(attn_weights, wv.out);
        wo.out = ttnn::add(ttnn::matmul(attn_out, wo.weight, false, true), wo.bias);

        residual1 = ttnn::add(x, wo.out);

        auto mean2 = ttnn::mean(residual1, -1, true);
        auto centered2 = ttnn::subtract(residual1, mean2);
        auto var2 = ttnn::mean(ttnn::multiply(centered2, centered2), -1, true);
        ln2.rstd = ttnn::rsqrt(ttnn::add(var2, ln2.eps), true);
        ln2.x_norm = ttnn::multiply(centered2, ln2.rstd);
        ln2.out = ttnn::add(ttnn::multiply(ln2.gamma, ln2.x_norm), ln2.beta);

        ffn.w1.out = ttnn::add(ttnn::matmul(ln2.out, ffn.w1.weight, false, true), ffn.w1.bias);
        ffn.gelu_out = ttnn::gelu(ffn.w1.out, true);
        ffn.w2.out = ttnn::add(ttnn::matmul(ffn.gelu_out, ffn.w2.weight, false, true), ffn.w2.bias);

        output = ttnn::add(residual1, ffn.w2.out);
        return &output;
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
// TransformerLayerMXP: Transformer with batched typecast for BFP8 FFN
// - Attention: BF16 (unchanged)
// - FFN: BFP8 with batched typecast via TypecastCache
// =============================================================================

} // namespace static_autograd
