#pragma once

#include "common.hpp"

namespace static_autograd {

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

    void sgd_step(float lr, float momentum = 0.0f, float weight_decay = 0.0f) {
        w1.sgd_step(lr, momentum, weight_decay);
        w2.sgd_step(lr, momentum, weight_decay);
    }
};

// =============================================================================
// Linear3DBFP8: BFP8 Linear layer with fused matmul+bias (ttnn::linear)
// Uses BFP8 weights, bias, and activations. BF16 gradients for stability.
// Expects BFP8 input (from LayerNormBFP8) - no typecast needed.
// Key insight: Mixed-type add (BFP8 + BF16) is 4x slower than same-type.
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
          gelu_out(make_zeros(ttnn::Shape({batch, seq, ffn_dim}), dev, ttnn::DataType::BFLOAT8_B)),
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

    void sgd_step(float lr, float momentum = 0.0f, float weight_decay = 0.0f) {
        w1.sgd_step(lr);  // Linear3DBFP8 doesn't use momentum/wd yet
        w2.sgd_step(lr);
    }
};

// =============================================================================
// FFNBFP8MXP: Feed-Forward Network with batched typecast
// Uses LinearBFP8MXP - backward uses pre-casted BF16 buffers
// =============================================================================


struct FFNBFP8MXP {
    LinearBFP8MXP w1;    // [dim, ffn_dim]
    LinearBFP8MXP w2;    // [ffn_dim, dim]

    // GELU intermediate buffers
    Tensor gelu_out;     // BFP8 (for forward)
    Tensor gelu_out_bf16; // BF16 cache (for backward)
    Tensor d_gelu_out;   // BF16 (gradient)

    FFNBFP8MXP(uint32_t batch, uint32_t seq, uint32_t dim, uint32_t ffn_dim, float init, MeshDevice& dev)
        : w1(batch, seq, dim, ffn_dim, init, dev),
          w2(batch, seq, ffn_dim, dim, init, dev),
          gelu_out(make_zeros(ttnn::Shape({batch, seq, ffn_dim}), dev, ttnn::DataType::BFLOAT8_B)),
          gelu_out_bf16(make_zeros(ttnn::Shape({batch, seq, ffn_dim}), dev)),
          d_gelu_out(make_zeros(ttnn::Shape({batch, seq, ffn_dim}), dev)) {}

    Value* forward(Graph& g, Value* x) {
#ifdef TRACY_ENABLE
        ZoneScopedN("FFNBFP8MXP_forward");
#endif
        auto* h = w1.forward(g, x);
        // GELU operates on BFP8
        h = gelu(g, h, &gelu_out, &d_gelu_out);
        return w2.forward(g, h);
    }

    void sgd_step(float lr, float momentum = 0.0f, float weight_decay = 0.0f) {
        w1.sgd_step(lr);  // LinearBFP8MXP doesn't use momentum/wd yet
        w2.sgd_step(lr);
    }

    // SGD step that only updates BF16 master weights (no typecast)
    void sgd_step_bf16_only(float lr, float momentum = 0.0f, float weight_decay = 0.0f) {
        w1.sgd_step_bf16_only(lr);
        w2.sgd_step_bf16_only(lr);
    }

    // Register all BFP8 activations for batch typecast before backward
    void register_caches(TypecastCache& cache) {
        w1.register_caches(cache);
        w2.register_caches(cache);
        // GELU output is BFP8 and used by w2 backward
        cache.register_pair(&gelu_out, &gelu_out_bf16);
    }

    // Register weight pairs for batched BF16→BFP8 conversion after SGD
    void register_weight_caches(WeightTypecastCache& cache) {
        w1.register_weight_caches(cache);
        w2.register_weight_caches(cache);
    }
};

// =============================================================================
// PersistentTransformerLayer: Single transformer layer for GPT-2
// Architecture: LN1 -> Attention -> Add -> LN2 -> FFN -> Add
// Uses simplified attention (no reshape/transpose for now)
// =============================================================================

} // namespace static_autograd
