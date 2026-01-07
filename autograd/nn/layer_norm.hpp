#pragma once

#include "common.hpp"

namespace static_autograd {

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
          out(make_zeros(ttnn::Shape({batch, seq, d}), dev, ttnn::DataType::BFLOAT8_B)),  // BFP8 output
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

} // namespace static_autograd
