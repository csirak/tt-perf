// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// FFN test using libtorch (CPU) in FP32 + BF16, with TDF1 outputs and op logs.

#include "tensordiff/core/serialize.hpp"
#include "tensordiff/core/torch_bridge.hpp"
#include "tensordiff/core/base_tensor.hpp"

#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using tensordiff::core::CpuTensor;
using tensordiff::core::OpType;

static void save_torch_tensor(const torch::Tensor& t, const std::string& path, std::string_view origin) {
    auto cpu = tensordiff::core::from_torch(t);
    tensordiff::core::save_tensor(cpu, path, origin);
}

static void record_op_torch(std::string_view name,
                            OpType type,
                            const std::vector<torch::Tensor>& inputs,
                            const std::vector<std::string>& input_names,
                            const torch::Tensor& output,
                            std::string_view origin,
                            std::string_view output_name,
                            std::string_view opts = {}) {
    std::vector<CpuTensor> cpu_inputs;
    cpu_inputs.reserve(inputs.size());
    for (const auto& t : inputs) {
        cpu_inputs.push_back(tensordiff::core::from_torch(t));
    }
    std::vector<const CpuTensor*> input_ptrs;
    input_ptrs.reserve(cpu_inputs.size());
    for (auto& t : cpu_inputs) {
        input_ptrs.push_back(&t);
    }
    std::vector<std::string_view> origins(input_ptrs.size(), origin);
    std::vector<std::string_view> names;
    names.reserve(input_names.size());
    for (const auto& n : input_names) {
        names.emplace_back(n);
    }
    auto out_cpu = tensordiff::core::from_torch(output);
    tensordiff::core::record_op_cpu(name, type, input_ptrs, origins, names, out_cpu, origin, output_name, opts);
}

static void run_ffn(torch::Tensor x,
                    torch::Tensor w1,
                    torch::Tensor b1,
                    torch::Tensor w2,
                    torch::Tensor b2,
                    const std::string& out_dir,
                    const std::string& origin,
                    bool record_ops) {
    std::filesystem::create_directories(out_dir);
    if (record_ops) {
        tensordiff::core::enable_op_recording(std::filesystem::path(out_dir) / "oplog");
    }

    // Forward
    auto h1 = torch::matmul(x, w1);
    if (record_ops) record_op_torch("ffn_w1_matmul", OpType::Binary, {x, w1}, {"ffn_input", "w1"}, h1, origin, "ffn_w1_matmul");

    auto h2 = h1 + b1;
    if (record_ops) record_op_torch("ffn_w1_bias", OpType::Binary, {h1, b1}, {"ffn_w1_matmul", "b1"}, h2, origin, "ffn_w1_bias");

    auto h3 = torch::gelu(h2);
    if (record_ops) record_op_torch("ffn_gelu", OpType::Unary, {h2}, {"ffn_w1_bias"}, h3, origin, "ffn_gelu");

    auto h4 = torch::matmul(h3, w2);
    if (record_ops) record_op_torch("ffn_w2_matmul", OpType::Binary, {h3, w2}, {"ffn_gelu", "w2"}, h4, origin, "ffn_w2_matmul");

    auto out = h4 + b2;
    if (record_ops) record_op_torch("ffn_w2_bias", OpType::Binary, {h4, b2}, {"ffn_w2_matmul", "b2"}, out, origin, "ffn_w2_bias");

    auto loss = out.sum();
    if (record_ops) record_op_torch("loss", OpType::Unary, {out}, {"ffn_w2_bias"}, loss, origin, "loss");

    loss.backward();

    // Save tensors (TDF1)
    save_torch_tensor(x, out_dir + "/input.bin", origin);
    save_torch_tensor(out, out_dir + "/output.bin", origin);
    save_torch_tensor(loss, out_dir + "/loss.bin", origin);

    if (x.grad().defined()) save_torch_tensor(x.grad(), out_dir + "/input_grad.bin", origin);
    if (w1.grad().defined()) save_torch_tensor(w1.grad(), out_dir + "/w1_grad.bin", origin);
    if (b1.grad().defined()) save_torch_tensor(b1.grad(), out_dir + "/b1_grad.bin", origin);
    if (w2.grad().defined()) save_torch_tensor(w2.grad(), out_dir + "/w2_grad.bin", origin);
    if (b2.grad().defined()) save_torch_tensor(b2.grad(), out_dir + "/b2_grad.bin", origin);

    save_torch_tensor(w1, out_dir + "/w1.bin", origin);
    save_torch_tensor(b1, out_dir + "/b1.bin", origin);
    save_torch_tensor(w2, out_dir + "/w2.bin", origin);
    save_torch_tensor(b2, out_dir + "/b2.bin", origin);
}

int main() {
    const uint32_t B = 1;
    const uint32_t S = 32;
    const uint32_t D = 64;
    const uint32_t F = 256;
    const float init_range = 0.5f;

    const char* out_env = std::getenv("TENSORDIFFGRAD_OUT_DIR");
    std::string out_root = out_env ? std::string(out_env) : "tensordiffgrad/outputs/ffn_torch";

    const bool record_ops = std::getenv("TENSORDIFF_OPLOG_DIR") != nullptr;

    torch::manual_seed(1234);
    auto base_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto base_x = torch::rand({B, S, D}, base_opts) * (2.0f * init_range) - init_range;
    auto base_w1 = torch::rand({D, F}, base_opts) * (2.0f * init_range) - init_range;
    auto base_b1 = torch::rand({1, 1, F}, base_opts) * (2.0f * init_range) - init_range;
    auto base_w2 = torch::rand({F, D}, base_opts) * (2.0f * init_range) - init_range;
    auto base_b2 = torch::rand({1, 1, D}, base_opts) * (2.0f * init_range) - init_range;

    // FP32 run
    {
        auto x = base_x.clone().requires_grad_(true);
        auto w1 = base_w1.clone().requires_grad_(true);
        auto b1 = base_b1.clone().requires_grad_(true);
        auto w2 = base_w2.clone().requires_grad_(true);
        auto b2 = base_b2.clone().requires_grad_(true);
        run_ffn(x, w1, b1, w2, b2, out_root + "/torch_fp32", "torch_fp32", record_ops);
    }

    // BF16 run (same init values cast to bf16)
    {
        auto x = base_x.to(torch::kBFloat16).clone().requires_grad_(true);
        auto w1 = base_w1.to(torch::kBFloat16).clone().requires_grad_(true);
        auto b1 = base_b1.to(torch::kBFloat16).clone().requires_grad_(true);
        auto w2 = base_w2.to(torch::kBFloat16).clone().requires_grad_(true);
        auto b2 = base_b2.to(torch::kBFloat16).clone().requires_grad_(true);
        run_ffn(x, w1, b1, w2, b2, out_root + "/torch_bf16", "torch_bf16", record_ops);
    }

    std::cout << "torch ffn done: " << out_root << "\n";
    return 0;
}
