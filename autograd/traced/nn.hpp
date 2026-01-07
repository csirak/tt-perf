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
#include <ttnn/operations/rand/rand.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <cmath>
#include <vector>

namespace traced {

using MeshDevice = tt::tt_metal::distributed::MeshDevice;

// ============================================================================
// Optimized Matmul Configuration for BF16 Training
// ============================================================================
// Settings from bench_gemm_bf16_top10.cpp (achieves 85 TFLOPS on 8K³):
// - HiFi2: ~70 TFLOPS theoretical peak (1.5x faster than HiFi4, sufficient precision)
// - packer_l1_acc = true: ALWAYS enable to reduce DRAM traffic
// - math_approx_mode = true: faster approximate math
// - ETH dispatch: use 8x8 grid (set USE_ETH_DISPATCH=1 at runtime)

// Global compute config - HiFi2 for BF16 training
inline const ttnn::WormholeComputeKernelConfig& get_bf16_compute_config() {
    static const ttnn::WormholeComputeKernelConfig config{
        .math_fidelity = MathFidelity::HiFi2,  // HiFi2 for training (not LoFi)
        .math_approx_mode = true,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = true,  // ALWAYS enable for performance
    };
    return config;
}

// Runtime matmul with optimized BF16 HiFi2 config
inline Tensor opt_matmul(const Tensor& a, const Tensor& b,
                         bool transpose_a = false, bool transpose_b = false) {
    return ttnn::matmul(a, b, transpose_a, transpose_b, std::nullopt,
                        ttnn::DataType::BFLOAT16, std::nullopt,
                        std::nullopt, get_bf16_compute_config());
}

// Seed counter for deterministic but unique seeds
static uint32_t g_seed_counter = 42;

// Helper to create tensor on device - use DRAM for persistent tensors (weights, buffers)
// L1 is only for intermediate computation tensors
inline Tensor make_full(ttnn::Shape shape, float value, MeshDevice& device) {
    return ttnn::full(shape, value, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
}

inline Tensor make_zeros(ttnn::Shape shape, MeshDevice& device) {
    return ttnn::zeros(shape, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
}

// Helper to create uniform random tensor in range [-std, +std]
// Approximates normal(0, std) for weight initialization
// Uses DRAM for persistent weight tensors
inline Tensor make_randn(ttnn::Shape shape, float std, MeshDevice& device) {
    // Uniform(-sqrt(3)*std, sqrt(3)*std) has same variance as normal(0, std)
    float range = std::sqrt(3.0f) * std;
    return ttnn::rand(shape, device, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT,
                      ttnn::types::DRAM_MEMORY_CONFIG, -range, range, g_seed_counter++);
}

// Helper to create ROW_MAJOR random tensor for embedding weights
// ttnn::embedding requires weight in ROW_MAJOR layout
// Uses DRAM for persistent weight tensors
inline Tensor make_embedding_weight(ttnn::Shape shape, float std, MeshDevice& device) {
    // Create random in TILE_LAYOUT first, then untilize to ROW_MAJOR
    float range = std::sqrt(3.0f) * std;
    auto tiled = ttnn::rand(shape, device, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT,
                            ttnn::types::DRAM_MEMORY_CONFIG, -range, range, g_seed_counter++);
    return ttnn::untilize(tiled);  // Use default memory config for output
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

    // init is now the std for random init (0.02 matches PyTorch GPT-2 default)
    TracedLinear(uint32_t in_f, uint32_t out_f, float init, MeshDevice& device)
        : weight(make_randn(ttnn::Shape({out_f, in_f}), init, device)),
          bias(make_zeros(ttnn::Shape({1, out_f}), device)),
          d_weight(make_zeros(ttnn::Shape({out_f, in_f}), device)),
          d_bias(make_zeros(ttnn::Shape({1, out_f}), device)),
          in_features(in_f),
          out_features(out_f) {}

    // Forward: out = x @ weight.T + bias
    void forward(const Tensor& x, Tensor& out) {
        out = ttnn::add(opt_matmul(x, weight, false, true), bias);
    }

    // Backward: compute gradients and propagate
    void backward(const Tensor& x, const Tensor& d_out, Tensor& d_input) {
        // d_weight = d_out.T @ x
        d_weight = opt_matmul(d_out, x, true, false);
        // d_bias = sum(d_out, dim=0)
        d_bias = ttnn::sum(d_out, 0, true);
        // d_input = d_out @ weight
        d_input = opt_matmul(d_out, weight);
    }

    // Backward without d_input (for first layer)
    void backward_no_input(const Tensor& x, const Tensor& d_out) {
        d_weight = opt_matmul(d_out, x, true, false);
        d_bias = ttnn::sum(d_out, 0, true);
    }

    // SGD update - use L1 for temporary scaled gradients
    void sgd_step(float lr) {
        auto scaled_dw = ttnn::multiply(d_weight, lr, std::nullopt, ttnn::L1_MEMORY_CONFIG);
        weight = ttnn::subtract(weight, scaled_dw);
        auto scaled_db = ttnn::multiply(d_bias, lr, std::nullopt, ttnn::L1_MEMORY_CONFIG);
        bias = ttnn::subtract(bias, scaled_db);
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
        scores = ttnn::multiply(opt_matmul(q, k, false, true), scale);
        // Apply causal mask
        scores_masked = ttnn::add(scores, causal_mask);
        // Softmax over last dimension
        traced::softmax(scores_masked, -1, attn_weights);
        // Output = attn_weights @ V
        output = opt_matmul(attn_weights, v);
    }

    // Backward pass - computes d_q, d_k, d_v given d_output, use L1 for intermediates
    void backward(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& d_out) {
        // d_attn = d_output @ V.T
        d_attn = opt_matmul(d_out, v, false, true);
        // d_V = attn_weights.T @ d_output
        d_v = opt_matmul(attn_weights, d_out, true, false);

        // Softmax backward
        traced::softmax_backward(d_attn, attn_weights, -1, d_scores);

        // Scale gradients - use L1 for temporary
        auto d_scores_scaled = ttnn::multiply(d_scores, scale);

        // d_Q = d_scores_scaled @ K
        d_q = opt_matmul(d_scores_scaled, k);
        // d_K = d_scores_scaled.T @ Q
        d_k = opt_matmul(d_scores_scaled, q, true, false);
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
    // mean is reduced (small), use L1
    void forward(const Tensor& x, Tensor& out) {
        // mean = mean(x, dim=-1, keepdim=True) - reduced, use L1
        auto mean = ttnn::mean(x, -1, true, ttnn::L1_MEMORY_CONFIG);

        // x_centered = x - mean
        auto x_centered = ttnn::subtract(x, mean);

        // var = mean(x_centered^2, dim=-1, keepdim=True)
        auto x_sq = ttnn::multiply(x_centered, x_centered);
        auto var = ttnn::mean(x_sq, -1, true);

        // rstd = 1 / sqrt(var + eps) - use fast approximate mode
        auto var_eps = ttnn::add(var, eps);
        rstd = ttnn::rsqrt(var_eps, true);

        // x_norm = x_centered * rstd
        x_norm = ttnn::multiply(x_centered, rstd);

        // out = gamma * x_norm + beta
        auto scaled = ttnn::multiply(gamma, x_norm);
        out = ttnn::add(scaled, beta);
    }

    // Backward: compute d_input, d_gamma, d_beta
    // All intermediates use L1 memory for minimal DRAM traffic
    void backward(const Tensor& x, const Tensor& d_out, Tensor& d_input) {
        // d_beta = sum(d_out, dims=[0,1], keepdim=True)
        auto sum0 = ttnn::sum(d_out, 0, true);
        d_beta = ttnn::sum(sum0, 1, true);

        // d_gamma = sum(d_out * x_norm, dims=[0,1], keepdim=True)
        auto d_out_x_norm = ttnn::multiply(d_out, x_norm);
        auto sum0_g = ttnn::sum(d_out_x_norm, 0, true);
        d_gamma = ttnn::sum(sum0_g, 1, true);

        // d_x_norm = d_out * gamma
        auto d_x_norm = ttnn::multiply(d_out, gamma);

        // LayerNorm backward formula (simplified):
        // d_x = rstd * (d_x_norm - mean(d_x_norm) - x_norm * mean(d_x_norm * x_norm))
        auto mean_d_x_norm = ttnn::mean(d_x_norm, -1, true);
        auto d_x_norm_x_norm = ttnn::multiply(d_x_norm, x_norm);
        auto mean_d_x_norm_x_norm = ttnn::mean(d_x_norm_x_norm, -1, true);

        auto diff1 = ttnn::subtract(d_x_norm, mean_d_x_norm);
        auto term2 = ttnn::multiply(x_norm, mean_d_x_norm_x_norm);
        auto diff2 = ttnn::subtract(diff1, term2);
        d_input = ttnn::multiply(rstd, diff2);
    }

    // SGD update - use L1 for temporary scaled gradients
    void sgd_step(float lr) {
        auto scaled_dg = ttnn::multiply(d_gamma, lr, std::nullopt, ttnn::L1_MEMORY_CONFIG);
        gamma = ttnn::subtract(gamma, scaled_dg);
        auto scaled_db = ttnn::multiply(d_beta, lr, std::nullopt, ttnn::L1_MEMORY_CONFIG);
        beta = ttnn::subtract(beta, scaled_db);
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
    // Use L1 for temporary computation, but weights stay in DRAM
    void sgd_step(float lr) {
        // d_weight is in TILE_LAYOUT, need to untilize for update
        auto d_weight_rm = ttnn::untilize(d_weight);
        auto scaled = ttnn::multiply(d_weight_rm, lr, std::nullopt, ttnn::L1_MEMORY_CONFIG);
        weight = ttnn::subtract(weight, scaled);  // weight stays in its original memory config
    }
};

}  // namespace traced
