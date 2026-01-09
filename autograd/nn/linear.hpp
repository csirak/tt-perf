#pragma once

#include "common.hpp"

namespace static_autograd {

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

    static bool use_fp32_acc() {
        return std::getenv("LINEAR_FP32_ACC") != nullptr;
    }

    Linear(uint32_t batch, uint32_t in_f, uint32_t out_f, float init, MeshDevice& dev,
           InitKind init_kind = InitKind::Randn)
        : weight(make_init_tensor(ttnn::Shape({out_f, in_f}), init, dev, init_kind)),
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
        mm_out = use_fp32_acc()
            ? matmul_fp32_acc(*x->data, weight, false, true)
            : ttnn::matmul(*x->data, weight, false, true);

        auto* mm_v = g.node(&mm_out, &d_mm_out);
        mm_v->parents = {x, w};

        mm_v->backward_fn = [x, w, this, mm_v]() {
            if (!mm_v->grad) return;
            const auto& dout = *mm_v->grad;

            // dx = dout @ weight (no transpose - weight is [out, in])
            if (x->requires_grad) {
                x->accumulate_grad(use_fp32_acc()
                    ? matmul_fp32_acc(dout, weight, false, false)
                    : ttnn::matmul(dout, weight));
            }
            // dweight = dout.T @ x
            if (w->requires_grad) {
                w->accumulate_grad(use_fp32_acc()
                    ? matmul_fp32_acc(dout, *x->data, true, false)
                    : ttnn::matmul(dout, *x->data, true, false));
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

    LinearReLU(uint32_t batch, uint32_t in_f, uint32_t out_f, float init, MeshDevice& dev,
               InitKind init_kind = InitKind::Randn)
        : weight(make_init_tensor(ttnn::Shape({out_f, in_f}), init, dev, init_kind)),
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


struct Linear3D {
    Tensor weight;      // [out_dim, in_dim]
    Tensor bias;        // [1, 1, out_dim]
    Tensor d_weight;
    Tensor d_bias;

    // Velocity buffers for momentum
    Tensor v_weight;
    Tensor v_bias;

    Tensor out;         // [B, S, out_dim]
    Tensor d_out;

    uint32_t in_dim;
    uint32_t out_dim;

    static bool use_fp32_acc() {
        return std::getenv("LINEAR_FP32_ACC") != nullptr;
    }

    Linear3D(uint32_t batch, uint32_t seq, uint32_t in_d, uint32_t out_d, float init, MeshDevice& dev,
             InitKind init_kind = InitKind::Randn)
        : weight(make_init_tensor(ttnn::Shape({out_d, in_d}), init, dev, init_kind)),
          bias(make_zeros(ttnn::Shape({1, 1, out_d}), dev)),
          d_weight(make_zeros(ttnn::Shape({out_d, in_d}), dev)),
          d_bias(make_zeros(ttnn::Shape({1, 1, out_d}), dev)),
          v_weight(make_zeros(ttnn::Shape({out_d, in_d}), dev)),
          v_bias(make_zeros(ttnn::Shape({1, 1, out_d}), dev)),
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
        Tensor mm = use_fp32_acc()
            ? matmul_fp32_acc(*x->data, weight, false, true)
            : ttnn::matmul(*x->data, weight, false, true);
        out = ttnn::add(mm, bias);

        auto* v = g.node(&out, &d_out);
        v->parents = {x, w, b};

        v->backward_fn = [x, w, b, this, v]() {
            if (!v->grad) return;
            const auto& dout = *v->grad;

            // d_input = dout @ weight
            if (x->requires_grad) {
                x->accumulate_grad(use_fp32_acc()
                    ? matmul_fp32_acc(dout, weight, false, false)
                    : ttnn::matmul(dout, weight));
            }
            // d_weight = sum over batch of dout.T @ x (seq reduced by matmul)
            if (w->requires_grad) {
                auto dw = use_fp32_acc()
                    ? matmul_fp32_acc(dout, *x->data, true, false)
                    : ttnn::matmul(dout, *x->data, true, false);
                auto dw_sum = ttnn::sum(dw, 0, false);
                w->accumulate_grad(dw_sum);
            }
            // d_bias = sum(dout, dims=[0,1])
            if (b->requires_grad) {
                auto sum0 = ttnn::sum(dout, 0, true);
                b->accumulate_grad(ttnn::sum(sum0, 1, true));
            }
        };

        return v;
    }

    void sgd_step(float lr, float momentum = 0.0f, float weight_decay = 0.0f) {
        if (momentum > 0.0f) {
            // v = momentum * v + grad + wd * weight
            v_weight = ttnn::add(
                ttnn::multiply(v_weight, momentum),
                ttnn::add(d_weight, ttnn::multiply(weight, weight_decay))
            );
            v_bias = ttnn::add(ttnn::multiply(v_bias, momentum), d_bias);
            weight = ttnn::subtract(weight, ttnn::multiply(v_weight, lr));
            bias = ttnn::subtract(bias, ttnn::multiply(v_bias, lr));
        } else if (weight_decay > 0.0f) {
            auto update_w = ttnn::add(d_weight, ttnn::multiply(weight, weight_decay));
            weight = ttnn::subtract(weight, ttnn::multiply(update_w, lr));
            bias = ttnn::subtract(bias, ttnn::multiply(d_bias, lr));
        } else {
            weight = ttnn::subtract(weight, ttnn::multiply(d_weight, lr));
            bias = ttnn::subtract(bias, ttnn::multiply(d_bias, lr));
        }
    }
};

// =============================================================================
// FFN: Feed-Forward Network with GELU activation
// Linear -> GELU -> Linear
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

    Linear3DBFP8(uint32_t batch, uint32_t seq, uint32_t in_d, uint32_t out_d, float init, MeshDevice& dev,
                 InitKind init_kind = InitKind::Randn)
        : weight(make_zeros(ttnn::Shape({out_d, in_d}), dev, ttnn::DataType::BFLOAT8_B)),
          bias(make_zeros(ttnn::Shape({1, 1, out_d}), dev, ttnn::DataType::BFLOAT8_B)),  // BFP8 bias!
          d_weight(make_zeros(ttnn::Shape({out_d, in_d}), dev)),
          d_bias(make_zeros(ttnn::Shape({1, 1, out_d}), dev)),
          weight_bf16(make_zeros(ttnn::Shape({out_d, in_d}), dev)),
          bias_bf16(make_zeros(ttnn::Shape({1, 1, out_d}), dev)),
          out(make_zeros(ttnn::Shape({batch, seq, out_d}), dev, ttnn::DataType::BFLOAT8_B)),
          d_out(make_zeros(ttnn::Shape({batch, seq, out_d}), dev)),
          in_dim(in_d),
          out_dim(out_d) {
        auto init_bf16 = make_init_tensor(ttnn::Shape({out_dim, in_dim}), init, dev, init_kind);
        weight_bf16 = init_bf16;
        weight = ttnn::typecast(init_bf16, ttnn::DataType::BFLOAT8_B);
    }

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
// LinearBFP8MXP: BFP8 Linear with pre-allocated BF16 caches for batched typecast
//
// Key difference from Linear3DBFP8:
// - Has out_bf16 and x_bf16_cache buffers
// - backward_fn uses these caches (NO inline typecast!)
// - TypecastCache fills these caches BEFORE backward pass
// =============================================================================


struct LinearBFP8MXP {
    // BFP8 weights (forward compute)
    Tensor weight;          // BFP8 [out_dim, in_dim]
    Tensor bias;            // BFP8 [1, 1, out_dim]

    // BF16 master weights (SGD)
    Tensor weight_bf16;
    Tensor bias_bf16;

    // Gradients (BF16)
    Tensor d_weight;
    Tensor d_bias;

    // Output buffers
    Tensor out;             // BFP8 forward output
    Tensor d_out;           // BF16 gradient

    // *** MXP: BF16 caches for backward (filled by TypecastCache before backward) ***
    Tensor out_bf16;        // BF16 cache of forward output
    Tensor x_bf16_cache;    // BF16 cache of input activation

    // Store input Value* for cache registration
    Value* cached_x = nullptr;

    uint32_t in_dim;
    uint32_t out_dim;

    static ttnn::WormholeComputeKernelConfig get_compute_config() {
        return ttnn::WormholeComputeKernelConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .math_approx_mode = false,
            .fp32_dest_acc_en = false,
            .packer_l1_acc = true,
        };
    }

    LinearBFP8MXP(uint32_t batch, uint32_t seq, uint32_t in_d, uint32_t out_d, float init, MeshDevice& dev,
                  InitKind init_kind = InitKind::Randn)
        : weight(make_zeros(ttnn::Shape({out_d, in_d}), dev, ttnn::DataType::BFLOAT8_B)),
          bias(make_zeros(ttnn::Shape({1, 1, out_d}), dev, ttnn::DataType::BFLOAT8_B)),
          weight_bf16(make_zeros(ttnn::Shape({out_d, in_d}), dev)),
          bias_bf16(make_zeros(ttnn::Shape({1, 1, out_d}), dev)),
          d_weight(make_zeros(ttnn::Shape({out_d, in_d}), dev)),
          d_bias(make_zeros(ttnn::Shape({1, 1, out_d}), dev)),
          out(make_zeros(ttnn::Shape({batch, seq, out_d}), dev, ttnn::DataType::BFLOAT8_B)),
          d_out(make_zeros(ttnn::Shape({batch, seq, out_d}), dev)),
          // MXP caches
          out_bf16(make_zeros(ttnn::Shape({batch, seq, out_d}), dev)),
          x_bf16_cache(make_zeros(ttnn::Shape({batch, seq, in_d}), dev)),
          in_dim(in_d),
          out_dim(out_d) {
        auto init_bf16 = make_init_tensor(ttnn::Shape({out_dim, in_dim}), init, dev, init_kind);
        weight_bf16 = init_bf16;
        weight = ttnn::typecast(init_bf16, ttnn::DataType::BFLOAT8_B);
    }

    Value* forward(Graph& g, Value* x) {
#ifdef TRACY_ENABLE
        ZoneScopedN("LinearBFP8MXP_forward");
#endif
        auto* w = g.leaf(&weight, &d_weight, true);
        auto* b = g.leaf(&bias, &d_bias, true);

        // Store input for cache registration
        cached_x = x;

        // Fused BFP8 linear: x @ weight.T + bias
        out = ttnn::linear(*x->data, weight, bias,
                           /*transpose_a=*/false, /*transpose_b=*/true,
                           /*memory_config=*/std::nullopt,
                           /*dtype=*/std::nullopt,
                           /*program_config=*/std::nullopt,
                           /*activation=*/std::nullopt,
                           /*compute_kernel_config=*/get_compute_config());

        auto* v = g.node(&out, &d_out);
        v->parents = {x, w, b};

        // Backward uses PRE-CASTED BF16 tensors (no typecast here!)
        v->backward_fn = [x, w, b, this, v]() {
#ifdef TRACY_ENABLE
            ZoneScopedN("LinearBFP8MXP_backward");
#endif
            if (!v->grad) return;

            // Use pre-casted out_bf16 (gradient already BF16 from d_out)
            const auto& dout_bf16 = *v->grad;  // d_out is already BF16

            // d_input = dout @ weight
            if (x->requires_grad) {
#ifdef TRACY_ENABLE
                ZoneScopedN("d_input_matmul");
#endif
                x->accumulate_grad(ttnn::matmul(dout_bf16, weight_bf16));
            }

            // d_weight = dout.T @ x (use pre-casted x_bf16_cache!)
            if (w->requires_grad) {
#ifdef TRACY_ENABLE
                ZoneScopedN("d_weight_compute");
#endif
                // *** MXP: Use cached BF16 input instead of inline typecast ***
                w->accumulate_grad(ttnn::matmul(dout_bf16, x_bf16_cache, true, false));
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
        ZoneScopedN("LinearBFP8MXP_sgd_step");
#endif
        weight_bf16 = ttnn::subtract(weight_bf16, ttnn::multiply(d_weight, lr));
        bias_bf16 = ttnn::subtract(bias_bf16, ttnn::multiply(d_bias, lr));

        // Cast back to BFP8 for next forward
        ttnn::typecast(weight_bf16, ttnn::DataType::BFLOAT8_B, std::nullopt, weight);
        ttnn::typecast(bias_bf16, ttnn::DataType::BFLOAT8_B, std::nullopt, bias);
    }

    // SGD step that only updates BF16 master weights (no typecast)
    // Use with WeightTypecastCache for batched BF16→BFP8 conversion
    void sgd_step_bf16_only(float lr) {
#ifdef TRACY_ENABLE
        ZoneScopedN("LinearBFP8MXP_sgd_step_bf16_only");
#endif
        weight_bf16 = ttnn::subtract(weight_bf16, ttnn::multiply(d_weight, lr));
        bias_bf16 = ttnn::subtract(bias_bf16, ttnn::multiply(d_bias, lr));
        // NO typecast here - done in batch by WeightTypecastCache
    }

    // Register this layer's tensors for batch typecast
    // Call after forward(), before backward()
    void register_caches(TypecastCache& cache) {
        // Register input activation: x (BFP8) -> x_bf16_cache
        if (cached_x && cached_x->data) {
            cache.register_pair(cached_x->data, &x_bf16_cache);
        }
        // Note: out is BFP8, but d_out is already BF16 (gradient buffer)
        // If we needed to convert out for backward, we'd register it too
    }

    // Register weight pairs for batched BF16→BFP8 conversion after SGD
    void register_weight_caches(WeightTypecastCache& cache) {
        cache.register_pair(&weight_bf16, &weight);
        cache.register_pair(&bias_bf16, &bias);
    }
};

// =============================================================================
// FFNBFP8: Feed-Forward Network with BFP8 matmuls
// Linear (BFP8) -> GELU -> Linear (BFP8)
// =============================================================================

} // namespace static_autograd
