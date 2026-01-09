#pragma once

#include "common.hpp"

namespace static_autograd {

// DyT: Dynamic Tanh (tanh(alpha * x)) with affine per-channel scale/shift.
// alpha is a learnable scalar; gamma/beta are per-channel.
struct DyT {
    // Parameters
    Tensor alpha;      // scalar
    Tensor gamma;      // [1, 1, D]
    Tensor beta;       // [1, 1, D]
    Tensor d_alpha;
    Tensor d_gamma;
    Tensor d_beta;

    // Velocity buffers for momentum
    Tensor v_alpha;
    Tensor v_gamma;
    Tensor v_beta;

    // Forward buffers
    Tensor out;
    Tensor d_out;
    Tensor tanh_out;   // tanh(alpha * x)

    uint32_t dim;

    DyT(uint32_t batch, uint32_t seq, uint32_t d, float alpha_init, MeshDevice& dev)
        : alpha(make_full(ttnn::Shape({1}), alpha_init, dev)),
          gamma(make_full(ttnn::Shape({1, 1, d}), 1.0f, dev)),
          beta(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          d_alpha(make_zeros(ttnn::Shape({1}), dev)),
          d_gamma(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          d_beta(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          v_alpha(make_zeros(ttnn::Shape({1}), dev)),
          v_gamma(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          v_beta(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          out(make_zeros(ttnn::Shape({batch, seq, d}), dev)),
          d_out(make_zeros(ttnn::Shape({batch, seq, d}), dev)),
          tanh_out(make_zeros(ttnn::Shape({batch, seq, d}), dev)),
          dim(d) {}

    Value* forward(Graph& g, Value* x) {
#ifdef TRACY_ENABLE
        ZoneScopedN("DyT_forward");
#endif
        auto* a = g.leaf(&alpha, &d_alpha, true);
        auto* gam = g.leaf(&gamma, &d_gamma, true);
        auto* bet = g.leaf(&beta, &d_beta, true);

        // tanh(alpha * x)
        auto scaled = ttnn::multiply(*x->data, alpha);
        tanh_out = ttnn::tanh(scaled);

        // out = gamma * tanh_out + beta
        out = ttnn::add(ttnn::multiply(gamma, tanh_out), beta);

        auto* v = g.node(&out, &d_out);
        v->parents = {x, a, gam, bet};

        v->backward_fn = [x, a, gam, bet, this, v]() {
            if (!v->grad) return;
            const auto& dout = *v->grad;

            // d_beta = sum(dout, dims=[0,1])
            if (bet->requires_grad) {
                auto sum0 = ttnn::sum(dout, 0, true);
                bet->accumulate_grad(ttnn::sum(sum0, 1, true));
            }

            // d_gamma = sum(dout * tanh_out, dims=[0,1])
            if (gam->requires_grad) {
                auto d_out_tanh = ttnn::multiply(dout, tanh_out);
                auto sum0 = ttnn::sum(d_out_tanh, 0, true);
                gam->accumulate_grad(ttnn::sum(sum0, 1, true));
            }

            // d_x_scaled = dout * gamma * (1 - tanh_out^2)
            auto d_x_scaled = ttnn::multiply(dout, gamma);
            auto tanh_sq = ttnn::multiply(tanh_out, tanh_out);
            auto one_minus_t2 = ttnn::subtract(ttnn::ones_like(tanh_sq), tanh_sq);
            d_x_scaled = ttnn::multiply(d_x_scaled, one_minus_t2);

            // dx = d_x_scaled * alpha
            if (x->requires_grad) {
                x->accumulate_grad(ttnn::multiply(d_x_scaled, alpha));
            }

            // d_alpha = sum(d_x_scaled * x)
            if (a->requires_grad) {
                auto d_alpha_full = ttnn::multiply(d_x_scaled, *x->data);
                auto sum0 = ttnn::sum(d_alpha_full, 0, true);
                auto sum01 = ttnn::sum(sum0, 1, true);
                auto sum012 = ttnn::sum(sum01, 2, true);
                auto d_alpha_scalar = ttnn::reshape(sum012, ttnn::Shape({1}));
                a->accumulate_grad(d_alpha_scalar);
            }
        };

        return v;
    }

    void sgd_step(float lr, float momentum = 0.0f, float weight_decay = 0.0f) {
        if (momentum > 0.0f) {
            // v = momentum * v + grad + wd * param (no wd on alpha/gamma/beta typically)
            v_alpha = ttnn::add(ttnn::multiply(v_alpha, momentum), d_alpha);
            v_gamma = ttnn::add(ttnn::multiply(v_gamma, momentum), d_gamma);
            v_beta = ttnn::add(ttnn::multiply(v_beta, momentum), d_beta);
            alpha = ttnn::subtract(alpha, ttnn::multiply(v_alpha, lr));
            gamma = ttnn::subtract(gamma, ttnn::multiply(v_gamma, lr));
            beta = ttnn::subtract(beta, ttnn::multiply(v_beta, lr));
        } else {
            alpha = ttnn::subtract(alpha, ttnn::multiply(d_alpha, lr));
            gamma = ttnn::subtract(gamma, ttnn::multiply(d_gamma, lr));
            beta = ttnn::subtract(beta, ttnn::multiply(d_beta, lr));
        }
    }
};

} // namespace static_autograd
