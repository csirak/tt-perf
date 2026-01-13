// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "norms.hpp"

#include "tensordiff/core/serialize.hpp"

#include <numeric>
#include <random>
#include <string>
#include <string_view>

namespace tensordiffgrad::core {

struct TransformerLayer {
    uint32_t bsz;
    uint32_t seq;
    uint32_t dim;
    uint32_t ffn_dim;
    float eps;
    NormModule ln1;
    NormModule ln2;

    Value* wq = nullptr;
    Value* wk = nullptr;
    Value* wv = nullptr;
    Value* wo = nullptr;
    Value* bq = nullptr;
    Value* bk = nullptr;
    Value* bv = nullptr;
    Value* bo = nullptr;

    Value* w1 = nullptr;
    Value* b1 = nullptr;
    Value* w2 = nullptr;
    Value* b2 = nullptr;

    Value* mask = nullptr;

    TransformerLayer(uint32_t b, uint32_t s, uint32_t d, uint32_t ffn_mult, NormKind nk, float eps_)
        : bsz(b),
          seq(s),
          dim(d),
          ffn_dim(d * ffn_mult),
          eps(eps_),
          ln1(nk, d, eps_),
          ln2(nk, d, eps_) {}

    static CpuTensor random_tensor(const std::vector<uint32_t>& shape, std::mt19937& rng, float scale) {
        std::uniform_real_distribution<float> dist(-scale, scale);
        const size_t numel = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<>());
        std::vector<float> vals(numel);
        for (size_t i = 0; i < numel; ++i) {
            vals[i] = dist(rng);
        }
        return CpuTensor::from_f32(std::move(vals), shape);
    }

    static CpuTensor causal_mask(uint32_t bsz, uint32_t seq) {
        std::vector<float> vals(static_cast<size_t>(bsz) * seq * seq, 0.0f);
        for (uint32_t b0 = 0; b0 < bsz; ++b0) {
            for (uint32_t i = 0; i < seq; ++i) {
                for (uint32_t j = i + 1; j < seq; ++j) {
                    vals[(static_cast<size_t>(b0) * seq + i) * seq + j] = -1e4f;
                }
            }
        }
        return CpuTensor::from_f32(std::move(vals), {bsz, seq, seq});
    }

    void init(Graph& g, std::mt19937& rng, float init_scale = 0.5f, std::string_view prefix = {}) {
        const std::string p = std::string(prefix);
        ln1.init(g, p + "ln1_", 1.0f);
        ln2.init(g, p + "ln2_", 1.0f);

        wq = g.leaf(random_tensor({dim, dim}, rng, init_scale), true, p + "wq");
        wk = g.leaf(random_tensor({dim, dim}, rng, init_scale), true, p + "wk");
        wv = g.leaf(random_tensor({dim, dim}, rng, init_scale), true, p + "wv");
        wo = g.leaf(random_tensor({dim, dim}, rng, init_scale), true, p + "wo");
        bq = g.leaf(random_tensor({1, 1, dim}, rng, init_scale), true, p + "bq");
        bk = g.leaf(random_tensor({1, 1, dim}, rng, init_scale), true, p + "bk");
        bv = g.leaf(random_tensor({1, 1, dim}, rng, init_scale), true, p + "bv");
        bo = g.leaf(random_tensor({1, 1, dim}, rng, init_scale), true, p + "bo");

        w1 = g.leaf(random_tensor({dim, ffn_dim}, rng, init_scale), true, p + "w1");
        b1 = g.leaf(random_tensor({1, 1, ffn_dim}, rng, init_scale), true, p + "b1");
        w2 = g.leaf(random_tensor({ffn_dim, dim}, rng, init_scale), true, p + "w2");
        b2 = g.leaf(random_tensor({1, 1, dim}, rng, init_scale), true, p + "b2");

        mask = g.leaf(causal_mask(bsz, seq), false, p + "mask");
    }

    Value* forward(Graph& g, Value* x, std::string_view prefix = {}) {
        auto tag = [&](std::string_view op) {
            if (prefix.empty()) return std::string(op);
            return std::string(prefix) + std::string(op);
        };

        auto ln1_out = ln1.forward(g, x, bsz, seq, tag("ln1"));
        auto q = linear(g, ln1_out, wq, bq, tag("wq"), bsz, seq, dim, dim);
        auto k = linear(g, ln1_out, wk, bk, tag("wk"), bsz, seq, dim, dim);
        auto v = linear(g, ln1_out, wv, bv, tag("wv"), bsz, seq, dim, dim);

        auto scores = matmul_qk(g, q, k, bsz, seq, dim, tag("attn_scores"));
        auto scaled = scale(g, scores, 1.0f / std::sqrt(static_cast<float>(dim)), tag("attn_scale"));
        auto masked = add(g, scaled, mask, tag("attn_mask"));
        auto attn = softmax(g, masked, bsz, seq, seq, tag("attn_weights"));
        auto attn_out = matmul_av(g, attn, v, bsz, seq, dim, tag("attn_out"));
        auto attn_proj = linear(g, attn_out, wo, bo, tag("wo"), bsz, seq, dim, dim);

        auto residual1 = add(g, x, attn_proj, tag("residual1"));
        auto ln2_out = ln2.forward(g, residual1, bsz, seq, tag("ln2"));
        auto ffn1 = linear(g, ln2_out, w1, b1, tag("ffn_w1"), bsz, seq, dim, ffn_dim);
        auto ffn_gelu = gelu(g, ffn1, tag("ffn_gelu"));
        auto ffn2 = linear(g, ffn_gelu, w2, b2, tag("ffn_w2"), bsz, seq, ffn_dim, dim);
        auto out = add(g, residual1, ffn2, tag("output"));
        return out;
    }

    Value* linear(Graph& g,
                  Value* x,
                  Value* w,
                  Value* b,
                  const std::string& tag,
                  uint32_t bsz,
                  uint32_t seq,
                  uint32_t in_dim,
                  uint32_t out_dim) {
        auto wx = matmul_3d_2d_op(g, x, w, bsz, seq, in_dim, out_dim, tag + "_matmul");
        auto out = add_bias(g, wx, b, bsz, seq, out_dim, tag + "_bias");
        return out;
    }

    void save_params(const std::string& dir, const std::string& prefix = "") const {
        using tensordiff::core::save_tensor;
        const std::string p = prefix.empty() ? "" : (prefix + "_");
        save_tensor(wq->data, dir + "/" + p + "wq.bin", "cpu");
        save_tensor(wk->data, dir + "/" + p + "wk.bin", "cpu");
        save_tensor(wv->data, dir + "/" + p + "wv.bin", "cpu");
        save_tensor(wo->data, dir + "/" + p + "wo.bin", "cpu");
        save_tensor(bq->data, dir + "/" + p + "bq.bin", "cpu");
        save_tensor(bk->data, dir + "/" + p + "bk.bin", "cpu");
        save_tensor(bv->data, dir + "/" + p + "bv.bin", "cpu");
        save_tensor(bo->data, dir + "/" + p + "bo.bin", "cpu");
        save_tensor(w1->data, dir + "/" + p + "w1.bin", "cpu");
        save_tensor(b1->data, dir + "/" + p + "b1.bin", "cpu");
        save_tensor(w2->data, dir + "/" + p + "w2.bin", "cpu");
        save_tensor(b2->data, dir + "/" + p + "b2.bin", "cpu");
        if (ln1.gamma) save_tensor(ln1.gamma->data, dir + "/" + p + "ln1_gamma.bin", "cpu");
        if (ln1.beta) save_tensor(ln1.beta->data, dir + "/" + p + "ln1_beta.bin", "cpu");
        if (ln2.gamma) save_tensor(ln2.gamma->data, dir + "/" + p + "ln2_gamma.bin", "cpu");
        if (ln2.beta) save_tensor(ln2.beta->data, dir + "/" + p + "ln2_beta.bin", "cpu");
        if (ln1.alpha) save_tensor(ln1.alpha->data, dir + "/" + p + "ln1_alpha.bin", "cpu");
        if (ln2.alpha) save_tensor(ln2.alpha->data, dir + "/" + p + "ln2_alpha.bin", "cpu");
    }
};

}  // namespace tensordiffgrad::core
