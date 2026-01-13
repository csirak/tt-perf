// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Torch random init for a single Transformer layer forward.
// Saves input/params and output (bf16 + f32), plus 2D slices for viewer.

#include "core/serialize.hpp"
#include "core/torch_bridge.hpp"

#include <torch/torch.h>
#include <torch/nn/functional/normalization.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

using tensordiff::core::from_torch;
using tensordiff::core::save_tensor;

static torch::Tensor layer_norm(const torch::Tensor& x,
                                const torch::Tensor& weight,
                                const torch::Tensor& bias,
                                double eps) {
    auto opts = torch::nn::functional::LayerNormFuncOptions({x.size(-1)})
                    .weight(weight)
                    .bias(bias)
                    .eps(eps);
    return torch::nn::functional::layer_norm(x, opts);
}

static torch::Tensor linear3d(const torch::Tensor& x,
                              const torch::Tensor& weight,
                              const torch::Tensor& bias) {
    // weight: [out_dim, in_dim], bias: [1,1,out_dim]
    auto y = torch::matmul(x, weight.t());
    return y + bias;
}

static torch::Tensor split_heads(const torch::Tensor& x, int64_t B, int64_t S, int64_t H, int64_t Dh) {
    return x.view({B, S, H, Dh}).permute({0, 2, 1, 3});
}

static torch::Tensor merge_heads(const torch::Tensor& x, int64_t B, int64_t S, int64_t H, int64_t Dh) {
    auto y = x.permute({0, 2, 1, 3});
    return y.reshape({B, S, H * Dh});
}

static torch::Tensor gelu_none(const torch::Tensor& x) {
    auto opts = torch::nn::functional::GELUFuncOptions().approximate("none");
    return torch::nn::functional::gelu(x, opts);
}

static torch::Tensor uniform_tensor(const std::vector<int64_t>& shape,
                                    const torch::TensorOptions& opts,
                                    float min_val,
                                    float max_val) {
    auto t = torch::rand(shape, opts);
    return t * (max_val - min_val) + min_val;
}

int main() {
    const char* out_env = std::getenv("TENSORDIFF_OUT_DIR");
    std::filesystem::path out_dir = out_env ? out_env : "tensordiff/outputs/transformer_layer_torch_rand";
    std::filesystem::create_directories(out_dir);

    int64_t B = 1, S = 64, D = 64, H = 1, FFN_MULT = 4;
    if (const char* env = std::getenv("TDF_B")) B = std::stoll(env);
    if (const char* env = std::getenv("TDF_S")) S = std::stoll(env);
    if (const char* env = std::getenv("TDF_D")) D = std::stoll(env);
    if (const char* env = std::getenv("TDF_H")) H = std::stoll(env);
    if (const char* env = std::getenv("TDF_FFN_MULT")) FFN_MULT = std::stoll(env);

    double eps = 1e-5;
    if (const char* env = std::getenv("TDF_EPS")) eps = std::stod(env);

    uint32_t seed = 0;
    if (const char* env = std::getenv("TDF_SEED")) seed = static_cast<uint32_t>(std::stoul(env));

    torch::manual_seed(seed);

    float init_min = -0.5f;
    float init_max = 0.5f;
    if (const char* env = std::getenv("TDF_MIN")) init_min = std::stof(env);
    if (const char* env = std::getenv("TDF_MAX")) init_max = std::stof(env);

    int64_t Dh = D / H;

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // Input
    auto x_f32 = uniform_tensor({B, S, D}, opts_f32, init_min, init_max);

    // LayerNorm params
    auto ln1_gamma_f32 = uniform_tensor({D}, opts_f32, init_min, init_max);
    auto ln1_beta_f32 = uniform_tensor({D}, opts_f32, init_min, init_max);
    auto ln2_gamma_f32 = uniform_tensor({D}, opts_f32, init_min, init_max);
    auto ln2_beta_f32 = uniform_tensor({D}, opts_f32, init_min, init_max);

    // Attention weights/biases
    auto wq_f32 = uniform_tensor({D, D}, opts_f32, init_min, init_max);
    auto wk_f32 = uniform_tensor({D, D}, opts_f32, init_min, init_max);
    auto wv_f32 = uniform_tensor({D, D}, opts_f32, init_min, init_max);
    auto wo_f32 = uniform_tensor({D, D}, opts_f32, init_min, init_max);

    auto bq_f32 = uniform_tensor({1, 1, D}, opts_f32, init_min, init_max);
    auto bk_f32 = uniform_tensor({1, 1, D}, opts_f32, init_min, init_max);
    auto bv_f32 = uniform_tensor({1, 1, D}, opts_f32, init_min, init_max);
    auto bo_f32 = uniform_tensor({1, 1, D}, opts_f32, init_min, init_max);

    // FFN weights/biases
    int64_t ffn_dim = D * FFN_MULT;
    auto w1_f32 = uniform_tensor({ffn_dim, D}, opts_f32, init_min, init_max);
    auto b1_f32 = uniform_tensor({1, 1, ffn_dim}, opts_f32, init_min, init_max);
    auto w2_f32 = uniform_tensor({D, ffn_dim}, opts_f32, init_min, init_max);
    auto b2_f32 = uniform_tensor({1, 1, D}, opts_f32, init_min, init_max);

    // FP32 forward
    auto ln1_f32 = layer_norm(x_f32, ln1_gamma_f32, ln1_beta_f32, eps);
    auto q_f32 = linear3d(ln1_f32, wq_f32, bq_f32);
    auto k_f32 = linear3d(ln1_f32, wk_f32, bk_f32);
    auto v_f32 = linear3d(ln1_f32, wv_f32, bv_f32);

    auto qh_f32 = split_heads(q_f32, B, S, H, Dh);
    auto kh_f32 = split_heads(k_f32, B, S, H, Dh);
    auto vh_f32 = split_heads(v_f32, B, S, H, Dh);

    auto scale = 1.0 / std::sqrt(static_cast<double>(Dh));
    auto scores_f32 = torch::matmul(qh_f32, kh_f32.transpose(-1, -2)) * scale;
    auto mask_f32 = torch::triu(torch::full({1, 1, S, S}, -1e9, opts_f32), 1);
    auto scores_masked_f32 = scores_f32 + mask_f32;
    auto attn_f32 = torch::softmax(scores_masked_f32, -1);
    auto attn_out_h_f32 = torch::matmul(attn_f32, vh_f32);
    auto attn_out_f32 = merge_heads(attn_out_h_f32, B, S, H, Dh);
    auto attn_proj_f32 = linear3d(attn_out_f32, wo_f32, bo_f32);
    auto residual1_f32 = x_f32 + attn_proj_f32;

    auto ln2_f32 = layer_norm(residual1_f32, ln2_gamma_f32, ln2_beta_f32, eps);
    auto ffn1_f32 = linear3d(ln2_f32, w1_f32, b1_f32);
    auto ffn_gelu_f32 = gelu_none(ffn1_f32);
    auto ffn2_f32 = linear3d(ffn_gelu_f32, w2_f32, b2_f32);
    auto out_f32 = residual1_f32 + ffn2_f32;

    // BF16 forward (weights cast from f32)
    auto x_bf16 = x_f32.to(torch::kBFloat16);
    auto ln1_gamma_bf16 = ln1_gamma_f32.to(torch::kBFloat16);
    auto ln1_beta_bf16 = ln1_beta_f32.to(torch::kBFloat16);
    auto ln2_gamma_bf16 = ln2_gamma_f32.to(torch::kBFloat16);
    auto ln2_beta_bf16 = ln2_beta_f32.to(torch::kBFloat16);

    auto wq_bf16 = wq_f32.to(torch::kBFloat16);
    auto wk_bf16 = wk_f32.to(torch::kBFloat16);
    auto wv_bf16 = wv_f32.to(torch::kBFloat16);
    auto wo_bf16 = wo_f32.to(torch::kBFloat16);
    auto bq_bf16 = bq_f32.to(torch::kBFloat16);
    auto bk_bf16 = bk_f32.to(torch::kBFloat16);
    auto bv_bf16 = bv_f32.to(torch::kBFloat16);
    auto bo_bf16 = bo_f32.to(torch::kBFloat16);

    auto w1_bf16 = w1_f32.to(torch::kBFloat16);
    auto b1_bf16 = b1_f32.to(torch::kBFloat16);
    auto w2_bf16 = w2_f32.to(torch::kBFloat16);
    auto b2_bf16 = b2_f32.to(torch::kBFloat16);

    auto ln1_bf16 = layer_norm(x_bf16, ln1_gamma_bf16, ln1_beta_bf16, eps);
    auto q_bf16 = linear3d(ln1_bf16, wq_bf16, bq_bf16);
    auto k_bf16 = linear3d(ln1_bf16, wk_bf16, bk_bf16);
    auto v_bf16 = linear3d(ln1_bf16, wv_bf16, bv_bf16);

    auto qh_bf16 = split_heads(q_bf16, B, S, H, Dh);
    auto kh_bf16 = split_heads(k_bf16, B, S, H, Dh);
    auto vh_bf16 = split_heads(v_bf16, B, S, H, Dh);

    auto scores_bf16 = torch::matmul(qh_bf16, kh_bf16.transpose(-1, -2)) * scale;
    auto mask_bf16 = torch::triu(torch::full({1, 1, S, S}, -1e9, torch::TensorOptions().dtype(torch::kBFloat16)), 1);
    auto scores_masked_bf16 = scores_bf16 + mask_bf16;
    auto attn_bf16 = torch::softmax(scores_masked_bf16, -1);
    auto attn_out_h_bf16 = torch::matmul(attn_bf16, vh_bf16);
    auto attn_out_bf16 = merge_heads(attn_out_h_bf16, B, S, H, Dh);
    auto attn_proj_bf16 = linear3d(attn_out_bf16, wo_bf16, bo_bf16);
    auto residual1_bf16 = x_bf16 + attn_proj_bf16;

    auto ln2_bf16 = layer_norm(residual1_bf16, ln2_gamma_bf16, ln2_beta_bf16, eps);
    auto ffn1_bf16 = linear3d(ln2_bf16, w1_bf16, b1_bf16);
    auto ffn_gelu_bf16 = gelu_none(ffn1_bf16);
    auto ffn2_bf16 = linear3d(ffn_gelu_bf16, w2_bf16, b2_bf16);
    auto out_bf16 = residual1_bf16 + ffn2_bf16;

    // Save inputs/params (bf16) for TTNN
    save_tensor(from_torch(x_bf16), (out_dir / "torch_tx_in.bin").string(), "torch");
    save_tensor(from_torch(out_bf16), (out_dir / "torch_tx_out.bin").string(), "torch");

    save_tensor(from_torch(ln1_gamma_bf16.view({1, 1, D})), (out_dir / "torch_ln1_gamma.bin").string(), "torch");
    save_tensor(from_torch(ln1_beta_bf16.view({1, 1, D})), (out_dir / "torch_ln1_beta.bin").string(), "torch");
    save_tensor(from_torch(ln2_gamma_bf16.view({1, 1, D})), (out_dir / "torch_ln2_gamma.bin").string(), "torch");
    save_tensor(from_torch(ln2_beta_bf16.view({1, 1, D})), (out_dir / "torch_ln2_beta.bin").string(), "torch");

    save_tensor(from_torch(wq_bf16), (out_dir / "torch_wq_weight.bin").string(), "torch");
    save_tensor(from_torch(wk_bf16), (out_dir / "torch_wk_weight.bin").string(), "torch");
    save_tensor(from_torch(wv_bf16), (out_dir / "torch_wv_weight.bin").string(), "torch");
    save_tensor(from_torch(wo_bf16), (out_dir / "torch_wo_weight.bin").string(), "torch");

    save_tensor(from_torch(bq_bf16), (out_dir / "torch_wq_bias.bin").string(), "torch");
    save_tensor(from_torch(bk_bf16), (out_dir / "torch_wk_bias.bin").string(), "torch");
    save_tensor(from_torch(bv_bf16), (out_dir / "torch_wv_bias.bin").string(), "torch");
    save_tensor(from_torch(bo_bf16), (out_dir / "torch_wo_bias.bin").string(), "torch");

    save_tensor(from_torch(w1_bf16), (out_dir / "torch_ffn_w1_weight.bin").string(), "torch");
    save_tensor(from_torch(b1_bf16), (out_dir / "torch_ffn_w1_bias.bin").string(), "torch");
    save_tensor(from_torch(w2_bf16), (out_dir / "torch_ffn_w2_weight.bin").string(), "torch");
    save_tensor(from_torch(b2_bf16), (out_dir / "torch_ffn_w2_bias.bin").string(), "torch");

    // Save FP32 versions
    save_tensor(from_torch(x_f32), (out_dir / "torch_tx_in_f32.bin").string(), "torch");
    save_tensor(from_torch(out_f32), (out_dir / "torch_tx_out_f32.bin").string(), "torch");

    save_tensor(from_torch(ln1_gamma_f32.view({1, 1, D})), (out_dir / "torch_ln1_gamma_f32.bin").string(), "torch");
    save_tensor(from_torch(ln1_beta_f32.view({1, 1, D})), (out_dir / "torch_ln1_beta_f32.bin").string(), "torch");
    save_tensor(from_torch(ln2_gamma_f32.view({1, 1, D})), (out_dir / "torch_ln2_gamma_f32.bin").string(), "torch");
    save_tensor(from_torch(ln2_beta_f32.view({1, 1, D})), (out_dir / "torch_ln2_beta_f32.bin").string(), "torch");

    save_tensor(from_torch(wq_f32), (out_dir / "torch_wq_weight_f32.bin").string(), "torch");
    save_tensor(from_torch(wk_f32), (out_dir / "torch_wk_weight_f32.bin").string(), "torch");
    save_tensor(from_torch(wv_f32), (out_dir / "torch_wv_weight_f32.bin").string(), "torch");
    save_tensor(from_torch(wo_f32), (out_dir / "torch_wo_weight_f32.bin").string(), "torch");

    save_tensor(from_torch(bq_f32), (out_dir / "torch_wq_bias_f32.bin").string(), "torch");
    save_tensor(from_torch(bk_f32), (out_dir / "torch_wk_bias_f32.bin").string(), "torch");
    save_tensor(from_torch(bv_f32), (out_dir / "torch_wv_bias_f32.bin").string(), "torch");
    save_tensor(from_torch(bo_f32), (out_dir / "torch_wo_bias_f32.bin").string(), "torch");

    save_tensor(from_torch(w1_f32), (out_dir / "torch_ffn_w1_weight_f32.bin").string(), "torch");
    save_tensor(from_torch(b1_f32), (out_dir / "torch_ffn_w1_bias_f32.bin").string(), "torch");
    save_tensor(from_torch(w2_f32), (out_dir / "torch_ffn_w2_weight_f32.bin").string(), "torch");
    save_tensor(from_torch(b2_f32), (out_dir / "torch_ffn_w2_bias_f32.bin").string(), "torch");

    // Save 2D slices for viewer (batch=0)
    auto in2d_bf16 = x_bf16.view({S, D});
    auto out2d_bf16 = out_bf16.view({S, D});
    auto in2d_f32 = x_f32.view({S, D});
    auto out2d_f32 = out_f32.view({S, D});

    save_tensor(from_torch(in2d_bf16), (out_dir / "torch_tx_in_2d.bin").string(), "torch");
    save_tensor(from_torch(out2d_bf16), (out_dir / "torch_tx_out_2d.bin").string(), "torch");
    save_tensor(from_torch(in2d_f32), (out_dir / "torch_tx_in_2d_f32.bin").string(), "torch");
    save_tensor(from_torch(out2d_f32), (out_dir / "torch_tx_out_2d_f32.bin").string(), "torch");

    std::cout << "# torch_transformer_layer_init: B=" << B << ", S=" << S << ", D=" << D
              << ", H=" << H << ", FFN_MULT=" << FFN_MULT << ", eps=" << eps
              << ", seed=" << seed << ", init=[" << init_min << ", " << init_max << "]\n";
    std::cout << "# wrote: " << out_dir.string() << "\n";

    return 0;
}
