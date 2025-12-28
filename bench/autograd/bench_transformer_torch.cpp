// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PyTorch/libtorch C++ Reference: Simple Transformer Layer
// CPU bf16 implementation for comparison with TTNN traced version.
//
// Build requires libtorch - typically built separately from tt-metal.
// This file is for reference/verification, not part of main benchmark suite.

#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstdlib>

// Simple transformer layer matching TTNN TracedTransformerLayer
// No LayerNorm, just attention + FFN with residuals
struct SimpleTransformerLayer {
    torch::Tensor wq, wk, wv, wo;  // Attention weights [D, D]
    torch::Tensor w1, w2;          // FFN weights
    int64_t dim;
    int64_t heads;
    int64_t head_dim;
    int64_t ffn_dim;
    float scale;

    SimpleTransformerLayer(int64_t d, int64_t h, int64_t ffn_mult = 4, float init_val = 0.01f)
        : dim(d), heads(h), head_dim(d / h), ffn_dim(d * ffn_mult),
          scale(1.0f / std::sqrt(static_cast<float>(head_dim)))
    {
        auto opts = torch::TensorOptions().dtype(torch::kBFloat16);

        // Attention weights
        wq = torch::full({d, d}, init_val, opts);
        wk = torch::full({d, d}, init_val, opts);
        wv = torch::full({d, d}, init_val, opts);
        wo = torch::full({d, d}, init_val, opts);

        // FFN weights
        w1 = torch::full({d, ffn_dim}, init_val, opts);
        w2 = torch::full({ffn_dim, d}, init_val, opts);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto B = x.size(0);
        auto S = x.size(1);
        auto D = x.size(2);

        // Q, K, V projections
        auto q_proj = torch::matmul(x, wq);
        auto k_proj = torch::matmul(x, wk);
        auto v_proj = torch::matmul(x, wv);

        // Reshape to [B, H, S, D/H]
        auto q = q_proj.view({B, S, heads, head_dim}).transpose(1, 2);
        auto k = k_proj.view({B, S, heads, head_dim}).transpose(1, 2);
        auto v = v_proj.view({B, S, heads, head_dim}).transpose(1, 2);

        // Attention scores
        auto scores = torch::matmul(q, k.transpose(-2, -1)) * scale;

        // Causal mask
        auto mask = torch::triu(torch::full({S, S}, -1e9f, x.options()), 1);
        scores = scores + mask;

        // Softmax
        auto attn = torch::softmax(scores, -1);

        // Attention output
        auto attn_out = torch::matmul(attn, v);

        // Merge heads: [B, H, S, D/H] -> [B, S, D]
        auto merged = attn_out.transpose(1, 2).contiguous().view({B, S, D});

        // Output projection
        auto attn_proj = torch::matmul(merged, wo);

        // Residual 1
        x = x + attn_proj;

        // FFN
        auto h1 = torch::matmul(x, w1);
        auto h1_relu = torch::relu(h1);
        auto ffn_out = torch::matmul(h1_relu, w2);

        // Residual 2
        return x + ffn_out;
    }
};

int main() {
    // Configuration from environment (matching TTNN benchmark)
    int64_t batch = 32, seq = 128, dim = 256, heads = 4, ffn_mult = 4;
    int64_t num_layers = 8;
    const float init_val = 0.01f;
    const int warmup = 3;
    const int iters = 10;

    if (const char* env = std::getenv("BENCH_BATCH")) batch = std::stol(env);
    if (const char* env = std::getenv("BENCH_SEQ")) seq = std::stol(env);
    if (const char* env = std::getenv("BENCH_DIM")) dim = std::stol(env);
    if (const char* env = std::getenv("BENCH_HEADS")) heads = std::stol(env);
    if (const char* env = std::getenv("BENCH_FFN_MULT")) ffn_mult = std::stol(env);
    if (const char* env = std::getenv("BENCH_LAYERS")) num_layers = std::stol(env);

    std::cout << "# PyTorch/libtorch Transformer Reference (CPU bf16)\n";
    std::cout << "# Config: batch=" << batch << ", seq=" << seq
              << ", dim=" << dim << ", heads=" << heads
              << ", ffn_mult=" << ffn_mult << "\n";
    std::cout << "# Layers: " << num_layers << "\n";
    std::cout << "# Warmup: " << warmup << ", Iterations: " << iters << "\n\n";

    // Create model
    SimpleTransformerLayer layer(dim, heads, ffn_mult, init_val);

    // Create input (bf16, CPU)
    auto x = torch::ones({batch, seq, dim}, torch::kBFloat16);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        auto y = x;
        for (int64_t l = 0; l < num_layers; ++l) {
            y = layer.forward(y);
        }
    }

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    torch::Tensor y;
    for (int i = 0; i < iters; ++i) {
        y = x;
        for (int64_t l = 0; l < num_layers; ++l) {
            y = layer.forward(y);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count() / iters;
    double per_layer_ms = total_ms / num_layers;

    std::cout << "## Results\n";
    std::cout << "Total time: " << total_ms << " ms\n";
    std::cout << "Per-layer: " << per_layer_ms << " ms\n\n";

    // Print first few output values for verification
    auto y_float = y.to(torch::kFloat32);
    std::cout << "## Output verification\n";
    std::cout << "Output shape: [" << y.size(0) << ", " << y.size(1) << ", " << y.size(2) << "]\n";
    std::cout << "First values: "
              << y_float[0][0][0].item<float>() << ", "
              << y_float[0][0][1].item<float>() << ", "
              << y_float[0][0][2].item<float>() << "\n";

    // CSV output for comparison
    std::cout << "\n# CSV: batch,seq,dim,heads,layers,total_ms,per_layer_ms\n";
    std::cout << "# " << batch << "," << seq << "," << dim << ","
              << heads << "," << num_layers << ","
              << total_ms << "," << per_layer_ms << "\n";

    return 0;
}
