// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// FFN test (CPU autograd) with op logging via TENSORDIFF_OPLOG_DIR.

#include "tensordiffgrad/core/autograd.hpp"
#include "tensordiffgrad/core/ops.hpp"

#include "tensordiff/core/serialize.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using tensordiffgrad::core::Graph;
using tensordiffgrad::core::Value;
using tensordiff::core::CpuTensor;
using tensordiff::core::DType;

static CpuTensor random_tensor(const std::vector<uint32_t>& shape, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    const size_t numel = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
    std::vector<float> vals(numel);
    for (size_t i = 0; i < numel; ++i) {
        vals[i] = dist(rng);
    }
    return CpuTensor::from_f32(std::move(vals), shape);
}

int main() {
    const uint32_t bsz = 1;
    const uint32_t seq = 32;
    const uint32_t dim = 64;
    const uint32_t ffn_dim = 256; // 4x
    const float init_scale = 0.5f;

    const char* out_env = std::getenv("TENSORDIFFGRAD_OUT_DIR");
    std::string out_dir = out_env ? std::string(out_env) : "tensordiffgrad/outputs/ffn";
    std::filesystem::create_directories(out_dir);

    Graph g;
    std::mt19937 rng(1234);

    // Inputs + params
    auto* x = g.leaf(random_tensor({bsz, seq, dim}, rng, init_scale), true, "ffn_input");
    auto* w1 = g.leaf(random_tensor({dim, ffn_dim}, rng, init_scale), true, "w1");
    auto* b1 = g.leaf(random_tensor({1, 1, ffn_dim}, rng, init_scale), true, "b1");
    auto* w2 = g.leaf(random_tensor({ffn_dim, dim}, rng, init_scale), true, "w2");
    auto* b2 = g.leaf(random_tensor({1, 1, dim}, rng, init_scale), true, "b2");

    // FFN forward: x -> (xW1 + b1) -> GELU -> (W2 + b2)
    auto* h1 = tensordiffgrad::core::matmul_3d_2d_op(g, x, w1, bsz, seq, dim, ffn_dim, "ffn_w1_matmul");
    auto* h2 = tensordiffgrad::core::add_bias(g, h1, b1, bsz, seq, ffn_dim, "ffn_w1_bias");
    auto* h3 = tensordiffgrad::core::gelu(g, h2, "ffn_gelu");
    auto* h4 = tensordiffgrad::core::matmul_3d_2d_op(g, h3, w2, bsz, seq, ffn_dim, dim, "ffn_w2_matmul");
    auto* out = tensordiffgrad::core::add_bias(g, h4, b2, bsz, seq, dim, "ffn_w2_bias");

    auto* loss = tensordiffgrad::core::sum_all(g, out, "loss");
    g.backward(loss);

    // Save a few tensors for inspection (TDF1)
    tensordiff::core::save_tensor(x->data, out_dir + "/input.bin", "cpu");
    tensordiff::core::save_tensor(out->data, out_dir + "/output.bin", "cpu");
    if (x->grad) tensordiff::core::save_tensor(*x->grad, out_dir + "/input_grad.bin", "cpu");

    std::cout << "ffn test done: " << out_dir << "\n";
    return 0;
}
