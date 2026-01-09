#pragma once

#include "common.hpp"
#include "dyt.hpp"
#include "adam.hpp"
#include <ttnn/operations/data_movement/permute/permute.hpp>

namespace static_autograd {

inline float attn_scale(uint32_t dim, uint32_t heads) {
    if (std::getenv("ATTN_SCALE_FULL_DIM")) {
        return 1.0f / std::sqrt(static_cast<float>(dim));
    }
    return 1.0f / std::sqrt(static_cast<float>(dim / heads));
}

inline Tensor split_heads(const Tensor& x, uint32_t batch, uint32_t seq, uint32_t heads, uint32_t head_dim) {
    auto reshaped = ttnn::reshape(x, ttnn::Shape({batch, seq, heads, head_dim}));
    ttnn::SmallVector<int64_t> dims = {0, 2, 1, 3};
    return ttnn::permute(reshaped, dims);
}

inline Tensor merge_heads(const Tensor& x, uint32_t batch, uint32_t seq, uint32_t heads, uint32_t head_dim) {
    ttnn::SmallVector<int64_t> dims = {0, 2, 1, 3};
    auto permuted = ttnn::permute(x, dims);
    return ttnn::reshape(permuted, ttnn::Shape({batch, seq, heads * head_dim}));
}

struct PersistentTransformerLayer {
    // Modules
    DyT ln1;
    DyT ln2;
    Linear3D wq, wk, wv, wo;
    FFN ffn;

    // Attention config
    float scale;
    uint32_t batch, seq, dim, heads, head_dim;

    // Forward buffers
    Tensor q_proj, k_proj, v_proj;           // [B, S, D]
    Tensor d_q_proj, d_k_proj, d_v_proj;
    Tensor q_heads, k_heads, v_heads;        // [B, H, S, Dh]
    Tensor attn_scores;                      // [B, H, S, S]
    Tensor d_attn_scores;
    Tensor attn_weights;                     // [B, H, S, S]
    Tensor d_attn_weights;
    Tensor attn_out_heads;                   // [B, H, S, Dh]
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
    Tensor causal_mask;                      // [1, 1, S, S]

    PersistentTransformerLayer(uint32_t b, uint32_t s, uint32_t d, uint32_t h,
                                uint32_t ffn_mult, float init, MeshDevice& dev,
                                float dyt_alpha = 1.0f)
        : ln1(b, s, d, dyt_alpha, dev),
          ln2(b, s, d, dyt_alpha, dev),
          wq(b, s, d, d, init, dev),
          wk(b, s, d, d, init, dev),
          wv(b, s, d, d, init, dev),
          wo(b, s, d, d, init, dev),
          ffn(b, s, d, d * ffn_mult, init, dev),
          scale(attn_scale(d, h)),
          batch(b), seq(s), dim(d), heads(h), head_dim(d / h),
          q_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          k_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          v_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_q_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_k_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_v_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          q_heads(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          k_heads(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          v_heads(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          attn_scores(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          d_attn_scores(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          attn_weights(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          d_attn_weights(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          attn_out_heads(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
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
        auto mask = make_full(ttnn::Shape({1, 1, s, s}), -1e9f, dev);
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

        // Multi-head attention
        q_heads = split_heads(*q_v->data, batch, seq, heads, head_dim);
        k_heads = split_heads(*k_v->data, batch, seq, heads, head_dim);
        v_heads = split_heads(*v_v->data, batch, seq, heads, head_dim);

        attn_scores = ttnn::multiply(matmul_fp32_acc(q_heads, k_heads, false, true), scale);
        auto scores_masked = ttnn::add(attn_scores, causal_mask);
        attn_weights = ttnn::softmax(scores_masked, -1, std::nullopt, get_softmax_compute_config(), true);

        attn_out_heads = matmul_fp32_acc(attn_weights, v_heads);
        attn_out = merge_heads(attn_out_heads, batch, seq, heads, head_dim);

        // Create node for attention output
        auto* attn_v = g.node(&attn_out, &d_attn_out);
        attn_v->parents = {q_v, k_v, v_v};

        // Attention backward (simplified - captures q,k,v for gradient computation)
        attn_v->backward_fn = [q_v, k_v, v_v, this, attn_v]() {
            if (!attn_v->grad) return;
            const auto& dout = *attn_v->grad;

            // Reshape upstream gradient to heads
            auto d_out_heads = split_heads(dout, batch, seq, heads, head_dim);

            // d_attn = d_out_heads @ V.T
            auto d_attn = matmul_fp32_acc(d_out_heads, v_heads, false, true);

            // d_V = attn_weights.T @ d_out_heads
            if (v_v->requires_grad) {
                auto d_v_heads = matmul_fp32_acc(attn_weights, d_out_heads, true, false);
                v_v->accumulate_grad(merge_heads(d_v_heads, batch, seq, heads, head_dim));
            }

            // Softmax backward
            auto dy_y = ttnn::multiply(d_attn, attn_weights);
            auto sum_dy_y = ttnn::sum(dy_y, -1, true, std::nullopt, get_fp32_acc_compute_config());
            auto d_scores = ttnn::multiply(attn_weights, ttnn::subtract(d_attn, sum_dy_y));
            auto d_scores_scaled = ttnn::multiply(d_scores, scale);

            // d_Q = d_scores_scaled @ K
            if (q_v->requires_grad) {
                auto d_q_heads = matmul_fp32_acc(d_scores_scaled, k_heads);
                q_v->accumulate_grad(merge_heads(d_q_heads, batch, seq, heads, head_dim));
            }
            // d_K = d_scores_scaled.T @ Q
            if (k_v->requires_grad) {
                auto d_k_heads = matmul_fp32_acc(d_scores_scaled, q_heads, true, false);
                k_v->accumulate_grad(merge_heads(d_k_heads, batch, seq, heads, head_dim));
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
        auto ln1_scaled = ttnn::multiply(x, ln1.alpha);
        ln1.tanh_out = ttnn::tanh(ln1_scaled);
        ln1.out = ttnn::add(ttnn::multiply(ln1.gamma, ln1.tanh_out), ln1.beta);

        wq.out = ttnn::add(ttnn::matmul(ln1.out, wq.weight, false, true), wq.bias);
        wk.out = ttnn::add(ttnn::matmul(ln1.out, wk.weight, false, true), wk.bias);
        wv.out = ttnn::add(ttnn::matmul(ln1.out, wv.weight, false, true), wv.bias);

        q_heads = split_heads(wq.out, batch, seq, heads, head_dim);
        k_heads = split_heads(wk.out, batch, seq, heads, head_dim);
        v_heads = split_heads(wv.out, batch, seq, heads, head_dim);

        attn_scores = ttnn::multiply(matmul_fp32_acc(q_heads, k_heads, false, true), scale);
        auto scores_masked = ttnn::add(attn_scores, causal_mask);
        attn_weights = ttnn::softmax(scores_masked, -1, std::nullopt, get_softmax_compute_config(), true);

        attn_out_heads = matmul_fp32_acc(attn_weights, v_heads);
        attn_out = merge_heads(attn_out_heads, batch, seq, heads, head_dim);
        wo.out = ttnn::add(ttnn::matmul(attn_out, wo.weight, false, true), wo.bias);

        residual1 = ttnn::add(x, wo.out);

        auto ln2_scaled = ttnn::multiply(residual1, ln2.alpha);
        ln2.tanh_out = ttnn::tanh(ln2_scaled);
        ln2.out = ttnn::add(ttnn::multiply(ln2.gamma, ln2.tanh_out), ln2.beta);

        ffn.w1.out = ttnn::add(ttnn::matmul(ln2.out, ffn.w1.weight, false, true), ffn.w1.bias);
        ffn.gelu_out = gelu_forward_tensor(ffn.w1.out);
        ffn.w2.out = ttnn::add(ttnn::matmul(ffn.gelu_out, ffn.w2.weight, false, true), ffn.w2.bias);

        output = ttnn::add(residual1, ffn.w2.out);
        return &output;
    }

    void sgd_step(float lr, float momentum = 0.0f, float weight_decay = 0.0f) {
        ln1.sgd_step(lr, momentum, weight_decay);
        ln2.sgd_step(lr, momentum, weight_decay);
        wq.sgd_step(lr, momentum, weight_decay);
        wk.sgd_step(lr, momentum, weight_decay);
        wv.sgd_step(lr, momentum, weight_decay);
        wo.sgd_step(lr, momentum, weight_decay);
        ffn.sgd_step(lr, momentum, weight_decay);
    }

    // Register all parameters with Adam optimizer
    void register_adam(Adam& adam, MeshDevice& dev) {
        // DyT params (no weight decay)
        ADAM_REGISTER_DYT(adam, ln1, dev);
        ADAM_REGISTER_DYT(adam, ln2, dev);
        // Attention projections
        ADAM_REGISTER_LINEAR3D(adam, wq, dev);
        ADAM_REGISTER_LINEAR3D(adam, wk, dev);
        ADAM_REGISTER_LINEAR3D(adam, wv, dev);
        ADAM_REGISTER_LINEAR3D(adam, wo, dev);
        // FFN
        ADAM_REGISTER_LINEAR3D(adam, ffn.w1, dev);
        ADAM_REGISTER_LINEAR3D(adam, ffn.w2, dev);
    }
};

// =============================================================================
// PersistentTransformerLayerLN: Transformer layer with LayerNorm (BF16)
// Used for parity/debug runs to compare against standard LN behavior.
// =============================================================================
struct PersistentTransformerLayerLN {
    LayerNorm ln1;
    LayerNorm ln2;
    Linear3D wq, wk, wv, wo;
    FFN ffn;

    float scale;
    uint32_t batch, seq, dim, heads, head_dim;

    Tensor q_proj, k_proj, v_proj;
    Tensor d_q_proj, d_k_proj, d_v_proj;
    Tensor q_heads, k_heads, v_heads;
    Tensor attn_scores, d_attn_scores;
    Tensor attn_weights, d_attn_weights;
    Tensor attn_out_heads;
    Tensor attn_out, d_attn_out;
    Tensor attn_proj, d_attn_proj;
    Tensor residual1, d_residual1;
    Tensor ffn_out, d_ffn_out;
    Tensor output, d_output;

    Tensor causal_mask;

    PersistentTransformerLayerLN(uint32_t b, uint32_t s, uint32_t d, uint32_t h,
                                 uint32_t ffn_mult, float init, MeshDevice& dev,
                                 float ln_eps)
        : ln1(b, s, d, ln_eps, dev),
          ln2(b, s, d, ln_eps, dev),
          wq(b, s, d, d, init, dev),
          wk(b, s, d, d, init, dev),
          wv(b, s, d, d, init, dev),
          wo(b, s, d, d, init, dev),
          ffn(b, s, d, d * ffn_mult, init, dev),
          scale(attn_scale(d, h)),
          batch(b), seq(s), dim(d), heads(h), head_dim(d / h),
          q_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          k_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          v_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_q_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_k_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_v_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          q_heads(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          k_heads(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          v_heads(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          attn_scores(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          d_attn_scores(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          attn_weights(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          d_attn_weights(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          attn_out_heads(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
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
        ZoneScopedN("TransformerLayerLN_forward");
#endif
        auto* ln1_v = ln1.forward(g, x);
        auto* q_v = wq.forward(g, ln1_v);
        auto* k_v = wk.forward(g, ln1_v);
        auto* v_v = wv.forward(g, ln1_v);

        q_heads = split_heads(*q_v->data, batch, seq, heads, head_dim);
        k_heads = split_heads(*k_v->data, batch, seq, heads, head_dim);
        v_heads = split_heads(*v_v->data, batch, seq, heads, head_dim);

        attn_scores = ttnn::multiply(matmul_fp32_acc(q_heads, k_heads, false, true), scale);
        auto scores_masked = ttnn::add(attn_scores, causal_mask);
        attn_weights = ttnn::softmax(scores_masked, -1, std::nullopt, get_softmax_compute_config(), true);
        attn_out_heads = matmul_fp32_acc(attn_weights, v_heads);
        attn_out = merge_heads(attn_out_heads, batch, seq, heads, head_dim);

        auto* attn_v = g.node(&attn_out, &d_attn_out);
        attn_v->parents = {q_v, k_v, v_v};

        attn_v->backward_fn = [q_v, k_v, v_v, this, attn_v]() {
            if (!attn_v->grad) return;
            const auto& dout = *attn_v->grad;

            auto d_out_heads = split_heads(dout, batch, seq, heads, head_dim);
            auto d_attn = matmul_fp32_acc(d_out_heads, v_heads, false, true);
            if (v_v->requires_grad) {
                auto d_v_heads = matmul_fp32_acc(attn_weights, d_out_heads, true, false);
                v_v->accumulate_grad(merge_heads(d_v_heads, batch, seq, heads, head_dim));
            }

            auto dy_y = ttnn::multiply(d_attn, attn_weights);
            auto sum_dy_y = ttnn::sum(dy_y, -1, true, std::nullopt, get_fp32_acc_compute_config());
            auto d_scores = ttnn::multiply(attn_weights, ttnn::subtract(d_attn, sum_dy_y));
            auto d_scores_scaled = ttnn::multiply(d_scores, scale);

            if (q_v->requires_grad) {
                auto d_q_heads = matmul_fp32_acc(d_scores_scaled, k_heads);
                q_v->accumulate_grad(merge_heads(d_q_heads, batch, seq, heads, head_dim));
            }
            if (k_v->requires_grad) {
                auto d_k_heads = matmul_fp32_acc(d_scores_scaled, q_heads, true, false);
                k_v->accumulate_grad(merge_heads(d_k_heads, batch, seq, heads, head_dim));
            }
        };

        auto* attn_proj_v = wo.forward(g, attn_v);
        auto* res1_v = add(g, x, attn_proj_v, &residual1, &d_residual1);
        auto* ln2_v = ln2.forward(g, res1_v);
        auto* ffn_v = ffn.forward(g, ln2_v);
        return add(g, res1_v, ffn_v, &output, &d_output);
    }

    Tensor* execute_forward(const Tensor& x) {
        ln1.execute_forward(x);

        wq.out = ttnn::add(ttnn::matmul(ln1.out, wq.weight, false, true), wq.bias);
        wk.out = ttnn::add(ttnn::matmul(ln1.out, wk.weight, false, true), wk.bias);
        wv.out = ttnn::add(ttnn::matmul(ln1.out, wv.weight, false, true), wv.bias);

        q_heads = split_heads(wq.out, batch, seq, heads, head_dim);
        k_heads = split_heads(wk.out, batch, seq, heads, head_dim);
        v_heads = split_heads(wv.out, batch, seq, heads, head_dim);

        attn_scores = ttnn::multiply(matmul_fp32_acc(q_heads, k_heads, false, true), scale);
        auto scores_masked = ttnn::add(attn_scores, causal_mask);
        attn_weights = ttnn::softmax(scores_masked, -1, std::nullopt, get_softmax_compute_config(), true);

        attn_out_heads = matmul_fp32_acc(attn_weights, v_heads);
        attn_out = merge_heads(attn_out_heads, batch, seq, heads, head_dim);
        wo.out = ttnn::add(ttnn::matmul(attn_out, wo.weight, false, true), wo.bias);

        residual1 = ttnn::add(x, wo.out);
        ln2.execute_forward(residual1);

        ffn.w1.out = ttnn::add(ttnn::matmul(ln2.out, ffn.w1.weight, false, true), ffn.w1.bias);
        ffn.gelu_out = gelu_forward_tensor(ffn.w1.out);
        ffn.w2.out = ttnn::add(ttnn::matmul(ffn.gelu_out, ffn.w2.weight, false, true), ffn.w2.bias);

        output = ttnn::add(residual1, ffn.w2.out);
        return &output;
    }

    void sgd_step(float lr, float momentum = 0.0f, float weight_decay = 0.0f) {
        ln1.sgd_step(lr, momentum, weight_decay);
        ln2.sgd_step(lr, momentum, weight_decay);
        wq.sgd_step(lr, momentum, weight_decay);
        wk.sgd_step(lr, momentum, weight_decay);
        wv.sgd_step(lr, momentum, weight_decay);
        wo.sgd_step(lr, momentum, weight_decay);
        ffn.sgd_step(lr, momentum, weight_decay);
    }

    // Register all parameters with Adam optimizer
    void register_adam(Adam& adam, MeshDevice& dev) {
        // LayerNorm params (no weight decay)
        ADAM_REGISTER_LAYERNORM(adam, ln1, dev);
        ADAM_REGISTER_LAYERNORM(adam, ln2, dev);
        // Attention projections
        ADAM_REGISTER_LINEAR3D(adam, wq, dev);
        ADAM_REGISTER_LINEAR3D(adam, wk, dev);
        ADAM_REGISTER_LINEAR3D(adam, wv, dev);
        ADAM_REGISTER_LINEAR3D(adam, wo, dev);
        // FFN
        ADAM_REGISTER_LINEAR3D(adam, ffn.w1, dev);
        ADAM_REGISTER_LINEAR3D(adam, ffn.w2, dev);
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
    Tensor q_heads, k_heads, v_heads;
    Tensor attn_scores, d_attn_scores;
    Tensor attn_weights, d_attn_weights;
    Tensor attn_out_heads;
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
          scale(attn_scale(d, h)),
          batch(b), seq(s), dim(d), heads(h), head_dim(d / h),
          q_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          k_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          v_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_q_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_k_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_v_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          q_heads(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          k_heads(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          v_heads(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          attn_scores(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          d_attn_scores(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          attn_weights(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          d_attn_weights(make_zeros(ttnn::Shape({b, h, s, s}), dev)),
          attn_out_heads(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
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

        q_heads = split_heads(*q_v->data, batch, seq, heads, head_dim);
        k_heads = split_heads(*k_v->data, batch, seq, heads, head_dim);
        v_heads = split_heads(*v_v->data, batch, seq, heads, head_dim);

        attn_scores = ttnn::multiply(matmul_fp32_acc(q_heads, k_heads, false, true), scale);
        auto scores_masked = ttnn::add(attn_scores, causal_mask);
        attn_weights = ttnn::softmax(scores_masked, -1, std::nullopt, get_softmax_compute_config(), true);

        attn_out_heads = matmul_fp32_acc(attn_weights, v_heads);
        attn_out = merge_heads(attn_out_heads, batch, seq, heads, head_dim);

        // Create node for attention output
        auto* attn_v = g.node(&attn_out, &d_attn_out);
        attn_v->parents = {q_v, k_v, v_v};

        // Attention backward
        attn_v->backward_fn = [q_v, k_v, v_v, this, attn_v]() {
            if (!attn_v->grad) return;
            const auto& dout = *attn_v->grad;

            auto d_out_heads = split_heads(dout, batch, seq, heads, head_dim);
            auto d_attn = matmul_fp32_acc(d_out_heads, v_heads, false, true);
            if (v_v->requires_grad) {
                auto d_v_heads = matmul_fp32_acc(attn_weights, d_out_heads, true, false);
                v_v->accumulate_grad(merge_heads(d_v_heads, batch, seq, heads, head_dim));
            }

            auto dy_y = ttnn::multiply(d_attn, attn_weights);
            auto sum_dy_y = ttnn::sum(dy_y, -1, true, std::nullopt, get_fp32_acc_compute_config());
            auto d_scores = ttnn::multiply(attn_weights, ttnn::subtract(d_attn, sum_dy_y));
            auto d_scores_scaled = ttnn::multiply(d_scores, scale);

            if (q_v->requires_grad) {
                auto d_q_heads = matmul_fp32_acc(d_scores_scaled, k_heads);
                q_v->accumulate_grad(merge_heads(d_q_heads, batch, seq, heads, head_dim));
            }
            if (k_v->requires_grad) {
                auto d_k_heads = matmul_fp32_acc(d_scores_scaled, q_heads, true, false);
                k_v->accumulate_grad(merge_heads(d_k_heads, batch, seq, heads, head_dim));
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

    void sgd_step(float lr, float momentum = 0.0f, float weight_decay = 0.0f) {
        ln1.sgd_step(lr, momentum, weight_decay);
        ln2.sgd_step(lr, momentum, weight_decay);
        wq.sgd_step(lr, momentum, weight_decay);
        wk.sgd_step(lr, momentum, weight_decay);
        wv.sgd_step(lr, momentum, weight_decay);
        wo.sgd_step(lr, momentum, weight_decay);
        ffn.sgd_step(lr, momentum, weight_decay);
    }
};

// =============================================================================
// TransformerLayerMXP: Transformer with batched typecast for BFP8 FFN
// - Attention: BF16 (unchanged)
// - FFN: BFP8 with batched typecast via TypecastCache
// =============================================================================

} // namespace static_autograd
