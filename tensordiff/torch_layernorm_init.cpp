// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Torch random init for LayerNorm. Saves inputs/params and outputs (bf16 + f32).

#include "core/serialize.hpp"
#include "core/torch_bridge.hpp"

#include <torch/torch.h>
#include <torch/nn/functional/normalization.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>

using tensordiff::core::CpuTensor;
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

int main() {
    const char* out_env = std::getenv("TENSORDIFF_OUT_DIR");
    std::filesystem::path out_dir = out_env ? out_env : "tensordiff/outputs/layernorm_torch_rand";
    std::filesystem::create_directories(out_dir);

    uint32_t B = 1, S = 32, D = 32;
    if (const char* env = std::getenv("TDF_B")) B = static_cast<uint32_t>(std::stoul(env));
    if (const char* env = std::getenv("TDF_S")) S = static_cast<uint32_t>(std::stoul(env));
    if (const char* env = std::getenv("TDF_D")) D = static_cast<uint32_t>(std::stoul(env));

    double eps = 1e-5;
    if (const char* env = std::getenv("TDF_EPS")) eps = std::stod(env);

    uint32_t seed = 0;
    if (const char* env = std::getenv("TDF_SEED")) seed = static_cast<uint32_t>(std::stoul(env));

    torch::manual_seed(seed);

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto x_f32 = torch::randn({static_cast<int64_t>(B), static_cast<int64_t>(S), static_cast<int64_t>(D)}, opts_f32);
    auto gamma_f32 = torch::randn({static_cast<int64_t>(D)}, opts_f32);
    auto beta_f32 = torch::randn({static_cast<int64_t>(D)}, opts_f32);

    auto out_f32 = layer_norm(x_f32, gamma_f32, beta_f32, eps);

    auto x_bf16 = x_f32.to(torch::kBFloat16);
    auto gamma_bf16 = gamma_f32.to(torch::kBFloat16);
    auto beta_bf16 = beta_f32.to(torch::kBFloat16);
    auto out_bf16 = layer_norm(x_bf16, gamma_bf16, beta_bf16, eps).to(torch::kBFloat16);

    // Save 3D tensors
    save_tensor(from_torch(x_bf16), (out_dir / "torch_ln_in.bin").string(), "torch");
    save_tensor(from_torch(out_bf16), (out_dir / "torch_ln_out.bin").string(), "torch");
    save_tensor(from_torch(gamma_bf16), (out_dir / "torch_ln_gamma.bin").string(), "torch");
    save_tensor(from_torch(beta_bf16), (out_dir / "torch_ln_beta.bin").string(), "torch");

    save_tensor(from_torch(x_f32), (out_dir / "torch_ln_in_f32.bin").string(), "torch");
    save_tensor(from_torch(out_f32), (out_dir / "torch_ln_out_f32.bin").string(), "torch");
    save_tensor(from_torch(gamma_f32), (out_dir / "torch_ln_gamma_f32.bin").string(), "torch");
    save_tensor(from_torch(beta_f32), (out_dir / "torch_ln_beta_f32.bin").string(), "torch");

    // Save 2D slices for viewer (batch=0)
    auto x2d_bf16 = x_bf16.view({static_cast<int64_t>(S), static_cast<int64_t>(D)});
    auto out2d_bf16 = out_bf16.view({static_cast<int64_t>(S), static_cast<int64_t>(D)});
    auto x2d_f32 = x_f32.view({static_cast<int64_t>(S), static_cast<int64_t>(D)});
    auto out2d_f32 = out_f32.view({static_cast<int64_t>(S), static_cast<int64_t>(D)});

    save_tensor(from_torch(x2d_bf16), (out_dir / "torch_ln_in_2d.bin").string(), "torch");
    save_tensor(from_torch(out2d_bf16), (out_dir / "torch_ln_out_2d.bin").string(), "torch");
    save_tensor(from_torch(x2d_f32), (out_dir / "torch_ln_in_2d_f32.bin").string(), "torch");
    save_tensor(from_torch(out2d_f32), (out_dir / "torch_ln_out_2d_f32.bin").string(), "torch");

    std::cout << "# torch_layernorm_init: B=" << B << ", S=" << S << ", D=" << D
              << ", eps=" << eps << ", seed=" << seed << "\n";
    std::cout << "# wrote: " << out_dir.string() << "\n";

    return 0;
}
