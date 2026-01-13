// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Recorder Test 1 (Torch C++): forward + backward dump of all tensors (TDF1).
// Loads TTNN outputs and replays in Torch (BF16 + FP32).

#include "tensordiff/core/serialize.hpp"
#include "tensordiff/core/torch_bridge.hpp"
#include "tensordiff/core/dtype.hpp"

#include <torch/torch.h>
#include <torch/nn/functional/activation.h>
#include <torch/nn/functional/normalization.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::string_view kLabel = "recorder_test1";

using tensordiff::core::CpuTensor;
using tensordiff::core::DType;
using tensordiff::core::load_tensor;
using tensordiff::core::save_tensor;

struct ManifestWriter {
    explicit ManifestWriter(const std::filesystem::path& path) {
        std::filesystem::create_directories(path.parent_path());
        file.open(path);
        if (file) {
            file << "name\tpath\tshape\tdtype\torigin\top_name\n";
        }
    }

    void write(const std::string& name,
               const std::string& path,
               const tensordiff::core::CpuTensor& t,
               std::string_view origin) {
        if (!file) return;
        const std::string rel = std::filesystem::path(path).filename().string();
        file << name << "\t" << rel << "\t" << shape_to_string(t.shape()) << "\t"
             << tensordiff::core::dtype_name(t.dtype()) << "\t" << origin << "\t" << kLabel << "\n";
    }

private:
    static std::string shape_to_string(const std::vector<uint32_t>& dims) {
        std::ostringstream oss;
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i) oss << "x";
            oss << dims[i];
        }
        return oss.str();
    }

    std::ofstream file;
};

void save_torch_tensor(const torch::Tensor& t,
                       const std::filesystem::path& path,
                       std::string_view origin,
                       ManifestWriter* manifest) {
    auto cpu = tensordiff::core::from_torch(t);
    save_tensor(cpu, path.string(), origin, kLabel);
    if (manifest) {
        manifest->write(path.filename().string(), path.string(), cpu, origin);
    }
}

void save_u32_tensor(const std::vector<uint32_t>& data,
                     const std::vector<uint32_t>& shape,
                     const std::filesystem::path& path,
                     std::string_view origin,
                     ManifestWriter* manifest) {
    CpuTensor t(shape, DType::kU32);
    if (t.numel() != data.size()) {
        throw std::runtime_error("save_u32_tensor: size mismatch");
    }
    std::memcpy(t.raw().data(), data.data(), data.size() * sizeof(uint32_t));
    save_tensor(t, path.string(), origin, kLabel);
    if (manifest) {
        manifest->write(path.filename().string(), path.string(), t, origin);
    }
}

std::vector<uint32_t> read_u32_vec(const CpuTensor& t) {
    if (t.dtype() != DType::kU32) {
        throw std::runtime_error("expected U32 tensor");
    }
    auto numel = t.numel();
    const auto* src = reinterpret_cast<const uint32_t*>(t.raw().data());
    return std::vector<uint32_t>(src, src + numel);
}

torch::Tensor to_torch_indices(const CpuTensor& t) {
    auto vals = read_u32_vec(t);
    std::vector<int64_t> out(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        out[i] = static_cast<int64_t>(vals[i]);
    }
    std::vector<int64_t> dims;
    dims.reserve(t.shape().size());
    for (auto d : t.shape()) {
        dims.push_back(static_cast<int64_t>(d));
    }
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto tensor = torch::from_blob(out.data(), dims, options).clone();
    return tensor;
}

torch::Tensor matmul_fp32(const torch::Tensor& a, const torch::Tensor& b, torch::Dtype out_dtype) {
    auto y = torch::matmul(a.to(torch::kFloat32), b.to(torch::kFloat32));
    return y.to(out_dtype);
}

torch::Tensor linear3d(const torch::Tensor& x,
                       const torch::Tensor& weight,
                       const torch::Tensor& bias,
                       torch::Dtype out_dtype) {
    auto y = matmul_fp32(x, weight.transpose(-1, -2), out_dtype);
    return (y + bias).to(out_dtype);
}

torch::Tensor layer_norm_manual(const torch::Tensor& x,
                                const torch::Tensor& gamma,
                                const torch::Tensor& beta,
                                double eps,
                                torch::Dtype out_dtype) {
    auto xf = x.to(torch::kFloat32);
    auto mean = xf.mean(-1, true);
    auto centered = xf - mean;
    auto var = (centered * centered).mean(-1, true);
    auto rstd = (var + eps).rsqrt();
    auto x_norm = centered * rstd;
    auto out = gamma.to(torch::kFloat32) * x_norm + beta.to(torch::kFloat32);
    return out.to(out_dtype);
}

torch::Tensor gelu_exact(const torch::Tensor& x, torch::Dtype out_dtype) {
    auto opts = torch::nn::functional::GELUFuncOptions().approximate("none");
    auto y = torch::nn::functional::gelu(x.to(torch::kFloat32), opts);
    return y.to(out_dtype);
}

void retain(torch::Tensor& t) {
    if (t.defined() && t.requires_grad()) {
        t.retain_grad();
    }
}

struct RunOutputs {
    torch::Tensor tok_embed;
    torch::Tensor pos_embed;
    torch::Tensor tok_plus_pos;
    torch::Tensor ln1_out;
    torch::Tensor q;
    torch::Tensor k;
    torch::Tensor v;
    torch::Tensor q_4d;
    torch::Tensor k_4d;
    torch::Tensor v_4d;
    torch::Tensor q_perm;
    torch::Tensor k_perm;
    torch::Tensor v_perm;
    torch::Tensor q_heads;
    torch::Tensor k_heads;
    torch::Tensor v_heads;
    torch::Tensor attn_scores_raw;
    torch::Tensor attn_scores_scaled;
    torch::Tensor attn_scores_masked;
    torch::Tensor attn_weights;
    torch::Tensor attn_out_heads;
    torch::Tensor attn_out_4d;
    torch::Tensor attn_out_perm;
    torch::Tensor attn_out;
    torch::Tensor attn_proj;
    torch::Tensor residual1;
    torch::Tensor ln2_out;
    torch::Tensor ffn_w1_out;
    torch::Tensor ffn_gelu;
    torch::Tensor ffn_w2_out;
    torch::Tensor output;
    torch::Tensor logits;
    torch::Tensor loss;
};

RunOutputs run_forward_backward(const std::filesystem::path& cpp_dir,
                               const std::filesystem::path& out_dir,
                               torch::Dtype dtype) {
    std::filesystem::create_directories(out_dir);
    ManifestWriter manifest(out_dir / "manifest.tsv");

    auto tokens = to_torch_indices(load_tensor((cpp_dir / "tokens_u32.bin").string()));
    auto pos = to_torch_indices(load_tensor((cpp_dir / "pos_u32.bin").string()));

    auto tok_weight = tensordiff::core::to_torch(load_tensor((cpp_dir / "tok_weight.bin").string()), dtype).requires_grad_(true);
    auto pos_weight = tensordiff::core::to_torch(load_tensor((cpp_dir / "pos_weight.bin").string()), dtype).requires_grad_(true);

    auto ln1_gamma = tensordiff::core::to_torch(load_tensor((cpp_dir / "ln1_gamma.bin").string()), dtype).requires_grad_(true);
    auto ln1_beta = tensordiff::core::to_torch(load_tensor((cpp_dir / "ln1_beta.bin").string()), dtype).requires_grad_(true);
    auto ln2_gamma = tensordiff::core::to_torch(load_tensor((cpp_dir / "ln2_gamma.bin").string()), dtype).requires_grad_(true);
    auto ln2_beta = tensordiff::core::to_torch(load_tensor((cpp_dir / "ln2_beta.bin").string()), dtype).requires_grad_(true);

    auto wq = tensordiff::core::to_torch(load_tensor((cpp_dir / "wq_weight.bin").string()), dtype).requires_grad_(true);
    auto wq_b = tensordiff::core::to_torch(load_tensor((cpp_dir / "wq_bias.bin").string()), dtype).requires_grad_(true);
    auto wk = tensordiff::core::to_torch(load_tensor((cpp_dir / "wk_weight.bin").string()), dtype).requires_grad_(true);
    auto wk_b = tensordiff::core::to_torch(load_tensor((cpp_dir / "wk_bias.bin").string()), dtype).requires_grad_(true);
    auto wv = tensordiff::core::to_torch(load_tensor((cpp_dir / "wv_weight.bin").string()), dtype).requires_grad_(true);
    auto wv_b = tensordiff::core::to_torch(load_tensor((cpp_dir / "wv_bias.bin").string()), dtype).requires_grad_(true);
    auto wo = tensordiff::core::to_torch(load_tensor((cpp_dir / "wo_weight.bin").string()), dtype).requires_grad_(true);
    auto wo_b = tensordiff::core::to_torch(load_tensor((cpp_dir / "wo_bias.bin").string()), dtype).requires_grad_(true);

    auto ffn_w1 = tensordiff::core::to_torch(load_tensor((cpp_dir / "ffn_w1_weight.bin").string()), dtype).requires_grad_(true);
    auto ffn_b1 = tensordiff::core::to_torch(load_tensor((cpp_dir / "ffn_w1_bias.bin").string()), dtype).requires_grad_(true);
    auto ffn_w2 = tensordiff::core::to_torch(load_tensor((cpp_dir / "ffn_w2_weight.bin").string()), dtype).requires_grad_(true);
    auto ffn_b2 = tensordiff::core::to_torch(load_tensor((cpp_dir / "ffn_w2_bias.bin").string()), dtype).requires_grad_(true);

    auto lm_head_w = tensordiff::core::to_torch(load_tensor((cpp_dir / "lm_head_weight.bin").string()), dtype).requires_grad_(true);
    auto lm_head_b = tensordiff::core::to_torch(load_tensor((cpp_dir / "lm_head_bias.bin").string()), dtype).requires_grad_(true);

    auto target = tensordiff::core::to_torch(load_tensor((cpp_dir / "target.bin").string()), dtype);

    const auto tok_plus_pos_ref = load_tensor((cpp_dir / "tok_plus_pos.bin").string());
    const auto q_heads_ref = load_tensor((cpp_dir / "q_heads.bin").string());
    if (tok_plus_pos_ref.shape().size() != 3 || q_heads_ref.shape().size() != 3) {
        throw std::runtime_error("expected 3D shapes for tok_plus_pos and q_heads");
    }
    const int64_t B = static_cast<int64_t>(tok_plus_pos_ref.shape()[0]);
    const int64_t S = static_cast<int64_t>(tok_plus_pos_ref.shape()[1]);
    const int64_t D = static_cast<int64_t>(tok_plus_pos_ref.shape()[2]);
    const int64_t H = static_cast<int64_t>(q_heads_ref.shape()[0]) / B;
    const int64_t Dh = static_cast<int64_t>(q_heads_ref.shape()[2]);
    if (H <= 0 || Dh <= 0 || D != H * Dh) {
        throw std::runtime_error("invalid head dims inferred from q_heads");
    }

    auto tok_embed = torch::embedding(tok_weight, tokens);
    auto pos_embed = torch::embedding(pos_weight, pos);
    auto tok_plus_pos = (tok_embed + pos_embed).to(dtype);

    auto ln1_out = layer_norm_manual(tok_plus_pos, ln1_gamma, ln1_beta, 1e-5, dtype);
    auto q = linear3d(ln1_out, wq, wq_b, dtype);
    auto k = linear3d(ln1_out, wk, wk_b, dtype);
    auto v = linear3d(ln1_out, wv, wv_b, dtype);

    auto q_4d = q.view({B, S, H, Dh});
    auto k_4d = k.view({B, S, H, Dh});
    auto v_4d = v.view({B, S, H, Dh});
    auto q_perm = q_4d.permute({0, 2, 1, 3});
    auto k_perm = k_4d.permute({0, 2, 1, 3});
    auto v_perm = v_4d.permute({0, 2, 1, 3});
    auto q_heads = q_perm.reshape({B * H, S, Dh});
    auto k_heads = k_perm.reshape({B * H, S, Dh});
    auto v_heads = v_perm.reshape({B * H, S, Dh});

    auto attn_scores_raw = matmul_fp32(q_heads, k_heads.transpose(-1, -2), dtype);
    auto scale = 1.0 / std::sqrt(static_cast<double>(Dh));
    auto attn_scores_scaled = (attn_scores_raw * scale).to(dtype);

    auto mask = torch::full({1, S, S}, -1e9, torch::TensorOptions().dtype(dtype));
    mask = torch::triu(mask, 1);
    auto attn_scores_masked = (attn_scores_scaled + mask).to(dtype);

    auto attn_weights = torch::softmax(attn_scores_masked.to(torch::kFloat32), -1).to(dtype);
    auto attn_out_heads = matmul_fp32(attn_weights, v_heads, dtype);
    auto attn_out_4d = attn_out_heads.view({B, H, S, Dh});
    auto attn_out_perm = attn_out_4d.permute({0, 2, 1, 3}).contiguous();
    auto attn_out = attn_out_perm.view({B, S, D});

    auto attn_proj = linear3d(attn_out, wo, wo_b, dtype);
    auto residual1 = (tok_plus_pos + attn_proj).to(dtype);

    auto ln2_out = layer_norm_manual(residual1, ln2_gamma, ln2_beta, 1e-5, dtype);
    auto ffn_w1_out = linear3d(ln2_out, ffn_w1, ffn_b1, dtype);
    auto ffn_gelu = gelu_exact(ffn_w1_out, dtype);
    auto ffn_w2_out = linear3d(ffn_gelu, ffn_w2, ffn_b2, dtype);
    auto output = (residual1 + ffn_w2_out).to(dtype);

    auto logits = linear3d(output, lm_head_w, lm_head_b, dtype);
    auto diff = (logits - target).to(dtype);
    auto loss = (diff * diff).mean().to(dtype);

    // retain grads
    retain(tok_embed);
    retain(pos_embed);
    retain(tok_plus_pos);
    retain(ln1_out);
    retain(q);
    retain(k);
    retain(v);
    retain(q_4d);
    retain(k_4d);
    retain(v_4d);
    retain(q_perm);
    retain(k_perm);
    retain(v_perm);
    retain(q_heads);
    retain(k_heads);
    retain(v_heads);
    retain(attn_scores_raw);
    retain(attn_scores_scaled);
    retain(attn_scores_masked);
    retain(attn_weights);
    retain(attn_out_heads);
    retain(attn_out_4d);
    retain(attn_out_perm);
    retain(attn_out);
    retain(attn_proj);
    retain(residual1);
    retain(ln2_out);
    retain(ffn_w1_out);
    retain(ffn_gelu);
    retain(ffn_w2_out);
    retain(output);
    retain(logits);
    retain(loss);

    loss.backward();

    const std::string origin = dtype == torch::kFloat32 ? "torch_fp32" : "torch_bf16";

    auto save_all = [&](const std::string& name, const torch::Tensor& t) {
        save_torch_tensor(t, out_dir / name, origin, &manifest);
    };

    // inputs + params
    save_u32_tensor(read_u32_vec(load_tensor((cpp_dir / "tokens_u32.bin").string())), {static_cast<uint32_t>(B), static_cast<uint32_t>(S)}, out_dir / "tokens_u32.bin", origin, &manifest);
    save_u32_tensor(read_u32_vec(load_tensor((cpp_dir / "pos_u32.bin").string())), {static_cast<uint32_t>(B), static_cast<uint32_t>(S)}, out_dir / "pos_u32.bin", origin, &manifest);

    save_all("tok_weight.bin", tok_weight);
    save_all("pos_weight.bin", pos_weight);
    save_all("ln1_gamma.bin", ln1_gamma);
    save_all("ln1_beta.bin", ln1_beta);
    save_all("ln2_gamma.bin", ln2_gamma);
    save_all("ln2_beta.bin", ln2_beta);

    save_all("wq_weight.bin", wq);
    save_all("wq_bias.bin", wq_b);
    save_all("wk_weight.bin", wk);
    save_all("wk_bias.bin", wk_b);
    save_all("wv_weight.bin", wv);
    save_all("wv_bias.bin", wv_b);
    save_all("wo_weight.bin", wo);
    save_all("wo_bias.bin", wo_b);

    save_all("ffn_w1_weight.bin", ffn_w1);
    save_all("ffn_w1_bias.bin", ffn_b1);
    save_all("ffn_w2_weight.bin", ffn_w2);
    save_all("ffn_w2_bias.bin", ffn_b2);

    save_all("lm_head_weight.bin", lm_head_w);
    save_all("lm_head_bias.bin", lm_head_b);

    // forward activations
    save_all("tok_embed.bin", tok_embed);
    save_all("pos_embed.bin", pos_embed);
    save_all("tok_plus_pos.bin", tok_plus_pos);
    save_all("ln1_out.bin", ln1_out);

    save_all("q.bin", q);
    save_all("k.bin", k);
    save_all("v.bin", v);
    save_all("q_4d.bin", q_4d);
    save_all("k_4d.bin", k_4d);
    save_all("v_4d.bin", v_4d);
    save_all("q_perm.bin", q_perm);
    save_all("k_perm.bin", k_perm);
    save_all("v_perm.bin", v_perm);
    save_all("q_heads.bin", q_heads);
    save_all("k_heads.bin", k_heads);
    save_all("v_heads.bin", v_heads);
    save_all("attn_scores_raw.bin", attn_scores_raw);
    save_all("attn_scores.bin", attn_scores_scaled);
    save_all("attn_causal_mask.bin", mask);
    save_all("attn_scores_masked.bin", attn_scores_masked);
    save_all("attn_weights.bin", attn_weights);
    save_all("attn_out_heads.bin", attn_out_heads);
    save_all("attn_out_4d.bin", attn_out_4d);
    save_all("attn_out_perm.bin", attn_out_perm);
    save_all("attn_out.bin", attn_out);
    save_all("attn_proj.bin", attn_proj);
    save_all("residual1.bin", residual1);

    save_all("ln2_out.bin", ln2_out);
    save_all("ffn_w1_out.bin", ffn_w1_out);
    save_all("ffn_gelu.bin", ffn_gelu);
    save_all("ffn_w2_out.bin", ffn_w2_out);
    save_all("output.bin", output);

    save_all("logits.bin", logits);
    save_all("loss.bin", loss.view({1}));
    save_all("target.bin", target);

    // grads
    if (tok_weight.grad().defined()) save_all("tok_weight_grad.bin", tok_weight.grad());
    if (pos_weight.grad().defined()) save_all("pos_weight_grad.bin", pos_weight.grad());
    if (tok_plus_pos.grad().defined()) save_all("tok_plus_pos_grad.bin", tok_plus_pos.grad());

    if (ln1_gamma.grad().defined()) save_all("ln1_gamma_grad.bin", ln1_gamma.grad());
    if (ln1_beta.grad().defined()) save_all("ln1_beta_grad.bin", ln1_beta.grad());
    if (ln2_gamma.grad().defined()) save_all("ln2_gamma_grad.bin", ln2_gamma.grad());
    if (ln2_beta.grad().defined()) save_all("ln2_beta_grad.bin", ln2_beta.grad());

    if (q.grad().defined()) save_all("q_grad.bin", q.grad());
    if (k.grad().defined()) save_all("k_grad.bin", k.grad());
    if (v.grad().defined()) save_all("v_grad.bin", v.grad());
    if (q_4d.grad().defined()) save_all("q_4d_grad.bin", q_4d.grad());
    if (k_4d.grad().defined()) save_all("k_4d_grad.bin", k_4d.grad());
    if (v_4d.grad().defined()) save_all("v_4d_grad.bin", v_4d.grad());
    if (q_perm.grad().defined()) save_all("q_perm_grad.bin", q_perm.grad());
    if (k_perm.grad().defined()) save_all("k_perm_grad.bin", k_perm.grad());
    if (v_perm.grad().defined()) save_all("v_perm_grad.bin", v_perm.grad());
    if (q_heads.grad().defined()) save_all("q_heads_grad.bin", q_heads.grad());
    if (k_heads.grad().defined()) save_all("k_heads_grad.bin", k_heads.grad());
    if (v_heads.grad().defined()) save_all("v_heads_grad.bin", v_heads.grad());
    if (attn_scores_raw.grad().defined()) save_all("attn_scores_grad.bin", attn_scores_raw.grad());
    if (attn_scores_scaled.grad().defined()) save_all("attn_scores_scaled_grad.bin", attn_scores_scaled.grad());
    if (attn_scores_masked.grad().defined()) save_all("attn_scores_masked_grad.bin", attn_scores_masked.grad());
    if (attn_weights.grad().defined()) save_all("attn_weights_grad.bin", attn_weights.grad());
    if (attn_out_heads.grad().defined()) save_all("attn_out_heads_grad.bin", attn_out_heads.grad());
    if (attn_out_4d.grad().defined()) save_all("attn_out_4d_grad.bin", attn_out_4d.grad());
    if (attn_out_perm.grad().defined()) save_all("attn_out_perm_grad.bin", attn_out_perm.grad());
    if (attn_out.grad().defined()) save_all("attn_out_grad.bin", attn_out.grad());
    if (attn_proj.grad().defined()) save_all("attn_proj_grad.bin", attn_proj.grad());
    if (residual1.grad().defined()) save_all("residual1_grad.bin", residual1.grad());

    if (wq.grad().defined()) save_all("wq_weight_grad.bin", wq.grad());
    if (wq_b.grad().defined()) save_all("wq_bias_grad.bin", wq_b.grad());
    if (wk.grad().defined()) save_all("wk_weight_grad.bin", wk.grad());
    if (wk_b.grad().defined()) save_all("wk_bias_grad.bin", wk_b.grad());
    if (wv.grad().defined()) save_all("wv_weight_grad.bin", wv.grad());
    if (wv_b.grad().defined()) save_all("wv_bias_grad.bin", wv_b.grad());
    if (wo.grad().defined()) save_all("wo_weight_grad.bin", wo.grad());
    if (wo_b.grad().defined()) save_all("wo_bias_grad.bin", wo_b.grad());

    if (ffn_w1_out.grad().defined()) save_all("ffn_w1_out_grad.bin", ffn_w1_out.grad());
    if (ffn_gelu.grad().defined()) save_all("ffn_gelu_grad.bin", ffn_gelu.grad());
    if (ffn_w2_out.grad().defined()) save_all("ffn_w2_out_grad.bin", ffn_w2_out.grad());
    if (output.grad().defined()) save_all("output_grad.bin", output.grad());

    if (ffn_w1.grad().defined()) save_all("ffn_w1_weight_grad.bin", ffn_w1.grad());
    if (ffn_b1.grad().defined()) save_all("ffn_w1_bias_grad.bin", ffn_b1.grad());
    if (ffn_w2.grad().defined()) save_all("ffn_w2_weight_grad.bin", ffn_w2.grad());
    if (ffn_b2.grad().defined()) save_all("ffn_w2_bias_grad.bin", ffn_b2.grad());

    if (lm_head_w.grad().defined()) save_all("lm_head_weight_grad.bin", lm_head_w.grad());
    if (lm_head_b.grad().defined()) save_all("lm_head_bias_grad.bin", lm_head_b.grad());
    if (logits.grad().defined()) save_all("logits_grad.bin", logits.grad());
    if (loss.grad().defined()) save_all("loss_grad.bin", loss.grad());

    return {tok_embed, pos_embed, tok_plus_pos, ln1_out, q, k, v, q_4d, k_4d, v_4d, q_perm, k_perm, v_perm,
            q_heads, k_heads, v_heads, attn_scores_raw, attn_scores_scaled, attn_scores_masked, attn_weights,
            attn_out_heads, attn_out_4d, attn_out_perm, attn_out, attn_proj, residual1, ln2_out, ffn_w1_out,
            ffn_gelu, ffn_w2_out, output, logits, loss};
}

} // namespace

int main() {
    const char* out_env = std::getenv("RECORDER_TEST1_OUT_DIR");
    std::filesystem::path base_dir = out_env
        ? std::filesystem::path(out_env)
        : std::filesystem::path("/home/howard/ttnn-perf/experiments/recorder_test1/outputs/recorder_test1");

    const char* cpp_env = std::getenv("RECORDER_TEST1_CPP_DIR");
    std::filesystem::path cpp_dir = cpp_env
        ? std::filesystem::path(cpp_env)
        : base_dir / "ttnn" / "step_0000";

    auto bf16_dir = base_dir / "torch_bf16" / "step_0000";
    auto fp32_dir = base_dir / "torch_fp32" / "step_0000";

    run_forward_backward(cpp_dir, bf16_dir, torch::kBFloat16);
    run_forward_backward(cpp_dir, fp32_dir, torch::kFloat32);

    std::cout << "recorder_test1: torch done (" << base_dir.string() << ")\n";
    return 0;
}
