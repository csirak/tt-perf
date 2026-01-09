#pragma once

#include "common.hpp"

namespace static_autograd {

struct TransformerLayerMXP {
    // Modules (attention stays BF16, FFN uses BFP8 with batched typecast)
    LayerNorm ln1;            // BF16 output (attention is BF16)
    LayerNormBFP8 ln2;        // BFP8 output (feeds directly into BFP8 FFN)
    Linear3D wq, wk, wv, wo;  // Attention projections stay BF16
    FFNBFP8MXP ffn;           // BFP8 FFN with MXP caching

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

    TransformerLayerMXP(uint32_t b, uint32_t s, uint32_t d, uint32_t h,
                        uint32_t ffn_mult, float init, MeshDevice& dev)
        : ln1(b, s, d, 1e-5f, dev),
          ln2(b, s, d, 1e-5f, dev),
          wq(b, s, d, d, init, dev),
          wk(b, s, d, d, init, dev),
          wv(b, s, d, d, init, dev),
          wo(b, s, d, d, init, dev),
          ffn(b, s, d, d * ffn_mult, init, dev),  // BFP8 FFN with MXP
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
        ZoneScopedN("TransformerLayerMXP_forward");
#endif
        // LN1
        auto* ln1_v = ln1.forward(g, x);

        // QKV projections (BF16)
        auto* q_v = wq.forward(g, ln1_v);
        auto* k_v = wk.forward(g, ln1_v);
        auto* v_v = wv.forward(g, ln1_v);

        // Simplified attention: scores = Q @ K.T * scale
        attn_scores = ttnn::multiply(matmul_fp32_acc(*q_v->data, *k_v->data, false, true), scale);

        // Apply causal mask
        auto scores_masked = ttnn::add(attn_scores, causal_mask);

        // Softmax
        attn_weights = ttnn::softmax(scores_masked, -1, std::nullopt, get_softmax_compute_config(), true);

        // attn_out = attn_weights @ V
        attn_out = matmul_fp32_acc(attn_weights, *v_v->data);

        // Create node for attention output
        auto* attn_v = g.node(&attn_out, &d_attn_out);
        attn_v->parents = {q_v, k_v, v_v};

        // Attention backward (unchanged - all BF16)
        attn_v->backward_fn = [q_v, k_v, v_v, this, attn_v]() {
            if (!attn_v->grad) return;
            const auto& dout = *attn_v->grad;

            auto d_attn = matmul_fp32_acc(dout, *v_v->data, false, true);
            if (v_v->requires_grad) {
                v_v->accumulate_grad(matmul_fp32_acc(attn_weights, dout, true, false));
            }

            auto dy_y = ttnn::multiply(d_attn, attn_weights);
            auto sum_dy_y = ttnn::sum(dy_y, -1, true);
            auto d_scores = ttnn::multiply(attn_weights, ttnn::subtract(d_attn, sum_dy_y));
            auto d_scores_scaled = ttnn::multiply(d_scores, scale);

            if (q_v->requires_grad) {
                q_v->accumulate_grad(matmul_fp32_acc(d_scores_scaled, *k_v->data));
            }
            if (k_v->requires_grad) {
                k_v->accumulate_grad(matmul_fp32_acc(d_scores_scaled, *q_v->data, true, false));
            }
        };

        // Output projection
        auto* attn_proj_v = wo.forward(g, attn_v);

        // Residual 1
        auto* res1_v = add(g, x, attn_proj_v, &residual1, &d_residual1);

        // LN2 (outputs BFP8 for FFN)
        auto* ln2_v = ln2.forward(g, res1_v);

        // BFP8 FFN (backward will use pre-casted BF16 from TypecastCache)
        auto* ffn_v = ffn.forward(g, ln2_v);

        // Residual 2
        return add(g, res1_v, ffn_v, &output, &d_output);
    }

    // Execute forward without graph construction (reuses buffers)
    Tensor* execute_forward(const Tensor& x) {
        // LN1 (BF16)
        auto mean_val = ttnn::mean(x, -1, true);
        auto x_centered = ttnn::subtract(x, mean_val);
        auto var = ttnn::mean(ttnn::multiply(x_centered, x_centered), -1, true);
        ln1.rstd = ttnn::rsqrt(ttnn::add(var, ln1.eps), true);
        ln1.x_norm = ttnn::multiply(x_centered, ln1.rstd);
        ln1.out = ttnn::add(ttnn::multiply(ln1.gamma, ln1.x_norm), ln1.beta);

        // QKV projections (BF16)
        wq.out = ttnn::add(ttnn::matmul(ln1.out, wq.weight, false, true), wq.bias);
        wk.out = ttnn::add(ttnn::matmul(ln1.out, wk.weight, false, true), wk.bias);
        wv.out = ttnn::add(ttnn::matmul(ln1.out, wv.weight, false, true), wv.bias);

        // Attention scores + mask + softmax
        attn_scores = ttnn::multiply(matmul_fp32_acc(wq.out, wk.out, false, true), scale);
        auto scores_masked = ttnn::add(attn_scores, causal_mask);
        attn_weights = ttnn::softmax(scores_masked, -1, std::nullopt, get_softmax_compute_config(), true);

        // Attention output
        attn_out = matmul_fp32_acc(attn_weights, wv.out);

        // Output projection
        wo.out = ttnn::add(ttnn::matmul(attn_out, wo.weight, false, true), wo.bias);

        // Residual 1
        residual1 = ttnn::add(x, wo.out);

        // LN2 (BFP8 output)
        auto mean2 = ttnn::mean(residual1, -1, true);
        auto centered2 = ttnn::subtract(residual1, mean2);
        auto var2 = ttnn::mean(ttnn::multiply(centered2, centered2), -1, true);
        ln2.rstd = ttnn::rsqrt(ttnn::add(var2, ln2.eps), true);
        ln2.x_norm = ttnn::multiply(centered2, ln2.rstd);
        auto scaled2 = ttnn::multiply(ln2.gamma, ln2.x_norm);
        ttnn::add(scaled2, ln2.beta, ttnn::DataType::BFLOAT8_B, std::nullopt, ln2.out);

        // BFP8 FFN
        ffn.w1.out = ttnn::linear(ln2.out, ffn.w1.weight, ffn.w1.bias,
                                  /*transpose_a=*/false, /*transpose_b=*/true,
                                  /*memory_config=*/std::nullopt,
                                  /*dtype=*/std::nullopt,
                                  /*program_config=*/std::nullopt,
                                  /*activation=*/std::nullopt,
                                  /*compute_kernel_config=*/LinearBFP8MXP::get_compute_config());
        ffn.gelu_out = gelu_forward_tensor(ffn.w1.out);
        ffn.w2.out = ttnn::linear(ffn.gelu_out, ffn.w2.weight, ffn.w2.bias,
                                  /*transpose_a=*/false, /*transpose_b=*/true,
                                  /*memory_config=*/std::nullopt,
                                  /*dtype=*/std::nullopt,
                                  /*program_config=*/std::nullopt,
                                  /*activation=*/std::nullopt,
                                  /*compute_kernel_config=*/LinearBFP8MXP::get_compute_config());

        // Residual 2
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

    // SGD step with BF16-only for FFN (attention stays inline)
    // Only FFN has BFP8 weights that need batched typecast
    void sgd_step_bf16_only(float lr) {
        // BF16 layers update normally (no separate typecast phase)
        ln1.sgd_step(lr);
        ln2.sgd_step(lr);
        wq.sgd_step(lr);
        wk.sgd_step(lr);
        wv.sgd_step(lr);
        wo.sgd_step(lr);
        // FFN: only update BF16 masters, skip typecast
        ffn.sgd_step_bf16_only(lr);
    }

    // Register FFN caches for batched typecast
    void register_caches(TypecastCache& cache) {
        ffn.register_caches(cache);
    }

    // Register FFN weight pairs for batched BF16→BFP8 conversion after SGD
    void register_weight_caches(WeightTypecastCache& cache) {
        ffn.register_weight_caches(cache);
    }
};

} // namespace static_autograd
