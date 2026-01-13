// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Simple transformer layer test (CPU autograd).
// Writes tensors in TDF1 format for torch parity + viewer.

#include "tensordiffgrad/core/transformer_layer.hpp"
#include "tensordiffgrad/core/ops.hpp"

#include "tensordiff/core/serialize.hpp"
#include "tensordiff/core/base_tensor.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using tensordiffgrad::core::Graph;
using tensordiffgrad::core::NormKind;
using tensordiffgrad::core::TransformerLayer;
using tensordiffgrad::core::Value;
using tensordiffgrad::core::sum_all;

static NormKind parse_norm() {
    const char* env = std::getenv("TENSORDIFFGRAD_NORM");
    std::string v = env ? std::string(env) : "rmsnorm";
    for (auto& c : v) c = static_cast<char>(std::tolower(c));
    if (v == "layernorm") return NormKind::LayerNorm;
    if (v == "dyt") return NormKind::DyT;
    return NormKind::RMSNorm;
}

static std::string norm_name(NormKind k) {
    switch (k) {
        case NormKind::LayerNorm: return "layernorm";
        case NormKind::RMSNorm: return "rmsnorm";
        case NormKind::DyT: return "dyt";
    }
    return "rmsnorm";
}

int main() {
    const uint32_t bsz = 1;
    const uint32_t seq = 4;
    const uint32_t dim = 32;
    const uint32_t ffn_mult = 4;
    const float eps = 1e-5f;

    const char* out_env = std::getenv("TENSORDIFFGRAD_OUT_DIR");
    std::string out_dir = out_env ? std::string(out_env) : "tensordiffgrad/outputs/run0";
    std::filesystem::create_directories(out_dir);

    const char* record_env = std::getenv("TENSORDIFF_OPLOG_DIR");
    if (record_env && record_env[0] != '\0') {
        tensordiff::core::enable_op_recording(record_env);
    }

    Graph g;
    std::mt19937 rng(1234);

    const char* layers_env = std::getenv("TENSORDIFFGRAD_LAYERS");
    const int layers = layers_env ? std::max(1, std::atoi(layers_env)) : 1;
    std::vector<TransformerLayer> stack;
    stack.reserve(static_cast<size_t>(layers));
    for (int i = 0; i < layers; ++i) {
        stack.emplace_back(bsz, seq, dim, ffn_mult, parse_norm(), eps);
        const std::string prefix = "layer" + std::to_string(i) + "_";
        stack.back().init(g, rng, 0.5f, prefix);
    }

    // Input
    auto x_cpu = TransformerLayer::random_tensor({bsz, seq, dim}, rng, 0.5f);
    auto* x = g.leaf(std::move(x_cpu), true, "layer0_input");

    // Forward + loss
    Value* h = x;
    for (int i = 0; i < layers; ++i) {
        const std::string prefix = "layer" + std::to_string(i) + "_";
        h = stack[static_cast<size_t>(i)].forward(g, h, prefix);
    }
    auto* out = h;
    auto* loss = sum_all(g, out, "loss");

    // Backward
    g.backward(loss);

    // Save tensors for torch parity + viewer
    using tensordiff::core::save_tensor;
    save_tensor(x->data, out_dir + "/input.bin", "cpu");
    save_tensor(out->data, out_dir + "/output.bin", "cpu");
    save_tensor(loss->data, out_dir + "/loss.bin", "cpu");

    if (x->grad) save_tensor(*x->grad, out_dir + "/input_grad.bin", "cpu");
    if (out->grad) save_tensor(*out->grad, out_dir + "/output_grad.bin", "cpu");
    if (loss->grad) save_tensor(*loss->grad, out_dir + "/loss_grad.bin", "cpu");

    if (layers == 1) {
        stack[0].save_params(out_dir);
    } else {
        for (int i = 0; i < layers; ++i) {
            const std::string prefix = "layer" + std::to_string(i);
            stack[static_cast<size_t>(i)].save_params(out_dir, prefix);
        }
    }

    if (layers == 1) {
        auto& layer = stack[0];
        if (layer.wq && layer.wq->grad) save_tensor(*layer.wq->grad, out_dir + "/wq_grad.bin", "cpu");
        if (layer.wk && layer.wk->grad) save_tensor(*layer.wk->grad, out_dir + "/wk_grad.bin", "cpu");
        if (layer.wv && layer.wv->grad) save_tensor(*layer.wv->grad, out_dir + "/wv_grad.bin", "cpu");
        if (layer.wo && layer.wo->grad) save_tensor(*layer.wo->grad, out_dir + "/wo_grad.bin", "cpu");
        if (layer.w1 && layer.w1->grad) save_tensor(*layer.w1->grad, out_dir + "/w1_grad.bin", "cpu");
        if (layer.w2 && layer.w2->grad) save_tensor(*layer.w2->grad, out_dir + "/w2_grad.bin", "cpu");
        if (layer.bq && layer.bq->grad) save_tensor(*layer.bq->grad, out_dir + "/bq_grad.bin", "cpu");
        if (layer.bk && layer.bk->grad) save_tensor(*layer.bk->grad, out_dir + "/bk_grad.bin", "cpu");
        if (layer.bv && layer.bv->grad) save_tensor(*layer.bv->grad, out_dir + "/bv_grad.bin", "cpu");
        if (layer.bo && layer.bo->grad) save_tensor(*layer.bo->grad, out_dir + "/bo_grad.bin", "cpu");
        if (layer.b1 && layer.b1->grad) save_tensor(*layer.b1->grad, out_dir + "/b1_grad.bin", "cpu");
        if (layer.b2 && layer.b2->grad) save_tensor(*layer.b2->grad, out_dir + "/b2_grad.bin", "cpu");

        if (layer.ln1.gamma && layer.ln1.gamma->grad) save_tensor(*layer.ln1.gamma->grad, out_dir + "/ln1_gamma_grad.bin", "cpu");
        if (layer.ln1.beta && layer.ln1.beta->grad) save_tensor(*layer.ln1.beta->grad, out_dir + "/ln1_beta_grad.bin", "cpu");
        if (layer.ln2.gamma && layer.ln2.gamma->grad) save_tensor(*layer.ln2.gamma->grad, out_dir + "/ln2_gamma_grad.bin", "cpu");
        if (layer.ln2.beta && layer.ln2.beta->grad) save_tensor(*layer.ln2.beta->grad, out_dir + "/ln2_beta_grad.bin", "cpu");
        if (layer.ln1.alpha && layer.ln1.alpha->grad) save_tensor(*layer.ln1.alpha->grad, out_dir + "/ln1_alpha_grad.bin", "cpu");
        if (layer.ln2.alpha && layer.ln2.alpha->grad) save_tensor(*layer.ln2.alpha->grad, out_dir + "/ln2_alpha_grad.bin", "cpu");
    }

    std::cout << "tensordiffgrad: done (" << out_dir << "), norm=" << norm_name(parse_norm()) << "\n";
    return 0;
}
