// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Torch random init for matmul inputs. Saves A/B inputs and torch output.

#include "core/serialize.hpp"
#include "core/torch_bridge.hpp"

#include <torch/torch.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>

using tensordiff::core::CpuTensor;
using tensordiff::core::from_torch;
using tensordiff::core::save_tensor;

int main() {
    const char* out_env = std::getenv("TENSORDIFF_OUT_DIR");
    std::filesystem::path out_dir = out_env ? out_env : "tensordiff/outputs/matmul_torch_rand";
    std::filesystem::create_directories(out_dir);

    uint32_t M = 64, K = 64, N = 64;
    if (const char* env = std::getenv("TDF_M")) M = static_cast<uint32_t>(std::stoul(env));
    if (const char* env = std::getenv("TDF_K")) K = static_cast<uint32_t>(std::stoul(env));
    if (const char* env = std::getenv("TDF_N")) N = static_cast<uint32_t>(std::stoul(env));

    uint32_t seed = 0;
    if (const char* env = std::getenv("TDF_SEED")) seed = static_cast<uint32_t>(std::stoul(env));

    torch::manual_seed(seed);

    float init_min = -0.5f;
    float init_max = 0.5f;
    if (const char* env = std::getenv("TDF_MIN")) init_min = std::stof(env);
    if (const char* env = std::getenv("TDF_MAX")) init_max = std::stof(env);

    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto a_f32 = torch::rand({static_cast<int64_t>(M), static_cast<int64_t>(K)}, opts_f32);
    auto b_f32 = torch::rand({static_cast<int64_t>(K), static_cast<int64_t>(N)}, opts_f32);
    a_f32 = a_f32 * (init_max - init_min) + init_min;
    b_f32 = b_f32 * (init_max - init_min) + init_min;
    auto out_f32 = torch::matmul(a_f32, b_f32);

    auto a_bf16 = a_f32.to(torch::kBFloat16);
    auto b_bf16 = b_f32.to(torch::kBFloat16);
    auto out_bf16 = torch::matmul(a_bf16, b_bf16).to(torch::kBFloat16);

    CpuTensor a_cpu = from_torch(a_bf16);
    CpuTensor b_cpu = from_torch(b_bf16);
    CpuTensor out_cpu = from_torch(out_bf16);

    CpuTensor a_f32_cpu = from_torch(a_f32);
    CpuTensor b_f32_cpu = from_torch(b_f32);
    CpuTensor out_f32_cpu = from_torch(out_f32);

    save_tensor(a_cpu, (out_dir / "torch_a.bin").string(), "torch");
    save_tensor(b_cpu, (out_dir / "torch_b.bin").string(), "torch");
    save_tensor(out_cpu, (out_dir / "torch_out.bin").string(), "torch");

    save_tensor(a_f32_cpu, (out_dir / "torch_a_f32.bin").string(), "torch");
    save_tensor(b_f32_cpu, (out_dir / "torch_b_f32.bin").string(), "torch");
    save_tensor(out_f32_cpu, (out_dir / "torch_out_f32.bin").string(), "torch");

    std::cout << "# torch_random_init: M=" << M << ", K=" << K << ", N=" << N
              << ", seed=" << seed << ", init=[" << init_min << ", " << init_max << "]\n";
    std::cout << "# wrote: " << out_dir.string() << "\n";

    return 0;
}
