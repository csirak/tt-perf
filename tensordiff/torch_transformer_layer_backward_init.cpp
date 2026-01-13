// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Torch backward init for a single Transformer layer.
// Saves inputs/params, upstream gradient, and intermediate/parameter grads (BF16).

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
    std::filesystem::path out_dir = out_env ? out_env : "tensordiff/outputs/transformer_layer_bwd_torch_rand";
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
    int64_t ffn_dim = D * FFN_MULT;

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // Inputs + params (init in FP32, then cast to BF16)
    auto x_f32 = uniform_tensor({B, S, D}, opts_f32, init_min, init_max);

    auto ln1_gamma_f32 = uniform_tensor({D}, opts_f32, init_min, init_max);
    auto ln1_beta_f32 = uniform_tensor({D}, opts_f32, init_min, init_max);
    auto ln2_gamma_f32 = uniform_tensor({D}, opts_f32, init_min, init_max);
    auto ln2_beta_f32 = uniform_tensor({D}, opts_f32, init_min, init_max);

    auto wq_f32 = uniform_tensor({D, D}, opts_f32, init_min, init_max);
    auto wk_f32 = uniform_tensor({D, D}, opts_f32, init_min, init_max);
    auto wv_f32 = uniform_tensor({D, D}, opts_f32, init_min, init_max);
    auto wo_f32 = uniform_tensor({D, D}, opts_f32, init_min, init_max);

    auto bq_f32 = uniform_tensor({1, 1, D}, opts_f32, init_min, init_max);
    auto bk_f32 = uniform_tensor({1, 1, D}, opts_f32, init_min, init_max);
    auto bv_f32 = uniform_tensor({1, 1, D}, opts_f32, init_min, init_max);
    auto bo_f32 = uniform_tensor({1, 1, D}, opts_f32, init_min, init_max);

    auto w1_f32 = uniform_tensor({ffn_dim, D}, opts_f32, init_min, init_max);
    auto b1_f32 = uniform_tensor({1, 1, ffn_dim}, opts_f32, init_min, init_max);
    auto w2_f32 = uniform_tensor({D, ffn_dim}, opts_f32, init_min, init_max);
    auto b2_f32 = uniform_tensor({1, 1, D}, opts_f32, init_min, init_max);

    auto x = x_f32.to(torch::kBFloat16).requires_grad_(true);

    auto ln1_gamma = ln1_gamma_f32.to(torch::kBFloat16).requires_grad_(true);
    auto ln1_beta = ln1_beta_f32.to(torch::kBFloat16).requires_grad_(true);
    auto ln2_gamma = ln2_gamma_f32.to(torch::kBFloat16).requires_grad_(true);
    auto ln2_beta = ln2_beta_f32.to(torch::kBFloat16).requires_grad_(true);

    auto wq = wq_f32.to(torch::kBFloat16).requires_grad_(true);
    auto wk = wk_f32.to(torch::kBFloat16).requires_grad_(true);
    auto wv = wv_f32.to(torch::kBFloat16).requires_grad_(true);
    auto wo = wo_f32.to(torch::kBFloat16).requires_grad_(true);

    auto bq = bq_f32.to(torch::kBFloat16).requires_grad_(true);
    auto bk = bk_f32.to(torch::kBFloat16).requires_grad_(true);
    auto bv = bv_f32.to(torch::kBFloat16).requires_grad_(true);
    auto bo = bo_f32.to(torch::kBFloat16).requires_grad_(true);

    auto w1 = w1_f32.to(torch::kBFloat16).requires_grad_(true);
    auto b1 = b1_f32.to(torch::kBFloat16).requires_grad_(true);
    auto w2 = w2_f32.to(torch::kBFloat16).requires_grad_(true);
    auto b2 = b2_f32.to(torch::kBFloat16).requires_grad_(true);

    // Forward (BF16)
    auto ln1 = layer_norm(x, ln1_gamma, ln1_beta, eps);
    ln1.retain_grad();

    auto q = linear3d(ln1, wq, bq);
    auto k = linear3d(ln1, wk, bk);
    auto v = linear3d(ln1, wv, bv);
    q.retain_grad();
    k.retain_grad();
    v.retain_grad();

    auto qh = split_heads(q, B, S, H, Dh);
    auto kh = split_heads(k, B, S, H, Dh);
    auto vh = split_heads(v, B, S, H, Dh);

    auto scale = 1.0 / std::sqrt(static_cast<double>(Dh));
    auto scores = torch::matmul(qh, kh.transpose(-1, -2)) * scale;
    scores.retain_grad();

    auto mask = torch::triu(torch::full({1, 1, S, S}, -1e9, torch::TensorOptions().dtype(torch::kBFloat16)), 1);
    auto scores_masked = scores + mask;
    auto attn = torch::softmax(scores_masked, -1);
    attn.retain_grad();

    auto attn_out_h = torch::matmul(attn, vh);
    auto attn_out = merge_heads(attn_out_h, B, S, H, Dh);
    attn_out.retain_grad();

    auto attn_proj = linear3d(attn_out, wo, bo);
    attn_proj.retain_grad();

    auto residual1 = x + attn_proj;
    residual1.retain_grad();

    auto ln2 = layer_norm(residual1, ln2_gamma, ln2_beta, eps);
    ln2.retain_grad();

    auto ffn1 = linear3d(ln2, w1, b1);
    ffn1.retain_grad();

    auto ffn_gelu = gelu_none(ffn1);
    ffn_gelu.retain_grad();

    auto ffn2 = linear3d(ffn_gelu, w2, b2);
    ffn2.retain_grad();

    auto out = residual1 + ffn2;
    out.retain_grad();

    // Upstream gradient
    auto d_out_f32 = uniform_tensor({B, S, D}, opts_f32, init_min, init_max);
    auto d_out = d_out_f32.to(torch::kBFloat16);

    // Backward
    torch::autograd::backward({out}, {d_out});

    // Softmax backward (manual, BF16) for binary-op comparison
    auto d_attn = attn.grad();
    auto dy_y = d_attn * attn;
    auto sum_dy_y = dy_y.sum(-1, true);
    auto d_scores_manual = attn * (d_attn - sum_dy_y);

    // Save inputs + params (BF16)
    save_tensor(from_torch(x), (out_dir / "torch_tx_in.bin").string(), "torch");
    save_tensor(from_torch(out), (out_dir / "torch_tx_out.bin").string(), "torch");
    save_tensor(from_torch(d_out), (out_dir / "torch_d_out.bin").string(), "torch");

    save_tensor(from_torch(attn), (out_dir / "torch_attn_weights.bin").string(), "torch");
    save_tensor(from_torch(d_attn), (out_dir / "torch_d_attn.bin").string(), "torch");
    save_tensor(from_torch(d_scores_manual), (out_dir / "torch_d_scores_manual.bin").string(), "torch");

    save_tensor(from_torch(ln1_gamma.view({1, 1, D})), (out_dir / "torch_ln1_gamma.bin").string(), "torch");
    save_tensor(from_torch(ln1_beta.view({1, 1, D})), (out_dir / "torch_ln1_beta.bin").string(), "torch");
    save_tensor(from_torch(ln2_gamma.view({1, 1, D})), (out_dir / "torch_ln2_gamma.bin").string(), "torch");
    save_tensor(from_torch(ln2_beta.view({1, 1, D})), (out_dir / "torch_ln2_beta.bin").string(), "torch");

    save_tensor(from_torch(wq), (out_dir / "torch_wq_weight.bin").string(), "torch");
    save_tensor(from_torch(wk), (out_dir / "torch_wk_weight.bin").string(), "torch");
    save_tensor(from_torch(wv), (out_dir / "torch_wv_weight.bin").string(), "torch");
    save_tensor(from_torch(wo), (out_dir / "torch_wo_weight.bin").string(), "torch");

    save_tensor(from_torch(bq), (out_dir / "torch_wq_bias.bin").string(), "torch");
    save_tensor(from_torch(bk), (out_dir / "torch_wk_bias.bin").string(), "torch");
    save_tensor(from_torch(bv), (out_dir / "torch_wv_bias.bin").string(), "torch");
    save_tensor(from_torch(bo), (out_dir / "torch_wo_bias.bin").string(), "torch");

    save_tensor(from_torch(w1), (out_dir / "torch_ffn_w1_weight.bin").string(), "torch");
    save_tensor(from_torch(b1), (out_dir / "torch_ffn_w1_bias.bin").string(), "torch");
    save_tensor(from_torch(w2), (out_dir / "torch_ffn_w2_weight.bin").string(), "torch");
    save_tensor(from_torch(b2), (out_dir / "torch_ffn_w2_bias.bin").string(), "torch");

    // Save intermediate grads (BF16)
    save_tensor(from_torch(out.grad().to(torch::kBFloat16)), (out_dir / "torch_d_output.bin").string(), "torch");
    save_tensor(from_torch(ffn2.grad().to(torch::kBFloat16)), (out_dir / "torch_d_ffn2.bin").string(), "torch");
    save_tensor(from_torch(ffn_gelu.grad().to(torch::kBFloat16)), (out_dir / "torch_d_ffn_gelu.bin").string(), "torch");
    save_tensor(from_torch(ffn1.grad().to(torch::kBFloat16)), (out_dir / "torch_d_ffn1.bin").string(), "torch");
    save_tensor(from_torch(ln2.grad().to(torch::kBFloat16)), (out_dir / "torch_d_ln2.bin").string(), "torch");
    save_tensor(from_torch(residual1.grad().to(torch::kBFloat16)), (out_dir / "torch_d_residual1.bin").string(), "torch");
    save_tensor(from_torch(attn_proj.grad().to(torch::kBFloat16)), (out_dir / "torch_d_attn_proj.bin").string(), "torch");
    save_tensor(from_torch(attn_out.grad().to(torch::kBFloat16)), (out_dir / "torch_d_attn_out.bin").string(), "torch");
    save_tensor(from_torch(attn.grad().to(torch::kBFloat16)), (out_dir / "torch_d_attn_weights.bin").string(), "torch");
    save_tensor(from_torch(scores.grad().to(torch::kBFloat16)), (out_dir / "torch_d_attn_scores.bin").string(), "torch");
    save_tensor(from_torch(q.grad().to(torch::kBFloat16)), (out_dir / "torch_d_q.bin").string(), "torch");
    save_tensor(from_torch(k.grad().to(torch::kBFloat16)), (out_dir / "torch_d_k.bin").string(), "torch");
    save_tensor(from_torch(v.grad().to(torch::kBFloat16)), (out_dir / "torch_d_v.bin").string(), "torch");
    save_tensor(from_torch(ln1.grad().to(torch::kBFloat16)), (out_dir / "torch_d_ln1.bin").string(), "torch");
    save_tensor(from_torch(x.grad().to(torch::kBFloat16)), (out_dir / "torch_d_x.bin").string(), "torch");

    // Save parameter grads (BF16)
    save_tensor(from_torch(wq.grad().to(torch::kBFloat16)), (out_dir / "torch_wq_weight_grad.bin").string(), "torch");
    save_tensor(from_torch(wk.grad().to(torch::kBFloat16)), (out_dir / "torch_wk_weight_grad.bin").string(), "torch");
    save_tensor(from_torch(wv.grad().to(torch::kBFloat16)), (out_dir / "torch_wv_weight_grad.bin").string(), "torch");
    save_tensor(from_torch(wo.grad().to(torch::kBFloat16)), (out_dir / "torch_wo_weight_grad.bin").string(), "torch");
    save_tensor(from_torch(bq.grad().to(torch::kBFloat16)), (out_dir / "torch_wq_bias_grad.bin").string(), "torch");
    save_tensor(from_torch(bk.grad().to(torch::kBFloat16)), (out_dir / "torch_wk_bias_grad.bin").string(), "torch");
    save_tensor(from_torch(bv.grad().to(torch::kBFloat16)), (out_dir / "torch_wv_bias_grad.bin").string(), "torch");
    save_tensor(from_torch(bo.grad().to(torch::kBFloat16)), (out_dir / "torch_wo_bias_grad.bin").string(), "torch");

    save_tensor(from_torch(w1.grad().to(torch::kBFloat16)), (out_dir / "torch_ffn_w1_weight_grad.bin").string(), "torch");
    save_tensor(from_torch(b1.grad().to(torch::kBFloat16)), (out_dir / "torch_ffn_w1_bias_grad.bin").string(), "torch");
    save_tensor(from_torch(w2.grad().to(torch::kBFloat16)), (out_dir / "torch_ffn_w2_weight_grad.bin").string(), "torch");
    save_tensor(from_torch(b2.grad().to(torch::kBFloat16)), (out_dir / "torch_ffn_w2_bias_grad.bin").string(), "torch");

    save_tensor(from_torch(ln1_gamma.grad().to(torch::kBFloat16)), (out_dir / "torch_ln1_gamma_grad.bin").string(), "torch");
    save_tensor(from_torch(ln1_beta.grad().to(torch::kBFloat16)), (out_dir / "torch_ln1_beta_grad.bin").string(), "torch");
    save_tensor(from_torch(ln2_gamma.grad().to(torch::kBFloat16)), (out_dir / "torch_ln2_gamma_grad.bin").string(), "torch");
    save_tensor(from_torch(ln2_beta.grad().to(torch::kBFloat16)), (out_dir / "torch_ln2_beta_grad.bin").string(), "torch");

    std::cout << "# torch_transformer_layer_backward_init: B=" << B << ", S=" << S
              << ", D=" << D << ", H=" << H << ", FFN_MULT=" << FFN_MULT
              << ", eps=" << eps << ", seed=" << seed
              << ", init=[" << init_min << ", " << init_max << "]\n";
    std::cout << "# wrote: " << out_dir.string() << "\n";
    return 0;
}
