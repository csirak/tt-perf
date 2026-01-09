// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GPT-2 reimplementation for parity testing (single layer, BF16).

#include "autograd/nn.hpp"
#include "autograd/utils/config.hpp"
#include "autograd/utils/tensor_io.hpp"

#include <ttnn/device.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace static_autograd;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;
using DispatchCoreType = tt::tt_metal::DispatchCoreType;

namespace {

struct DeviceGuard {
    std::shared_ptr<MeshDevice> device;

    DeviceGuard()
        : device([]() {
            bool use_worker = std::getenv("USE_WORKER_DISPATCH") != nullptr;
            return MeshDevice::create_unit_mesh(
                0,
                DEFAULT_L1_SMALL_SIZE,
                0,
                1,
                use_worker ? DispatchCoreConfig{}
                           : DispatchCoreConfig{DispatchCoreType::ETH}
            );
        }()) {
        auto grid = device->compute_with_storage_grid_size();
        fmt::print("# Compute grid: {}x{} = {} cores\n", grid.x, grid.y, grid.x * grid.y);
    }

    ~DeviceGuard() {
        tt::tt_metal::distributed::Finish(device->mesh_command_queue());
        ttnn::close_device(*device);
    }

    MeshDevice& get() { return *device; }
};

struct Config {
    std::string data_dir = "/home/howard/ttnn-perf/experiments/gpt2_reimpl/data";
    std::string outputs_dir = "/home/howard/ttnn-perf/experiments/gpt2_reimpl/outputs";
    uint32_t batch = 32;
    uint32_t seq = 32;
    uint32_t dim = 128;
    uint32_t heads = 4;
    uint32_t ffn_mult = 4;
    uint32_t vocab = 160;
    uint32_t steps = 1000;
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 0.01f;
    uint32_t log_every = 50;
    uint32_t seed = 1;
    float init_std = 0.02f;
    float ln_eps = 1e-5f;
    uint32_t dump_outputs = 1;
};

Config load_config(const std::string& path) {
    Config cfg;
    auto kv = load_kv_file(path);
    if (kv.empty()) {
        fmt::print("# Config not found: {} (using defaults)\n", path);
        return cfg;
    }
    get_str(kv, "data_dir", cfg.data_dir);
    get_str(kv, "outputs_dir", cfg.outputs_dir);
    get_u32(kv, "batch_size", cfg.batch);
    get_u32(kv, "seq", cfg.seq);
    get_u32(kv, "dim", cfg.dim);
    get_u32(kv, "heads", cfg.heads);
    get_u32(kv, "ffn_mult", cfg.ffn_mult);
    get_u32(kv, "vocab", cfg.vocab);
    get_u32(kv, "steps", cfg.steps);
    get_f32(kv, "lr", cfg.lr);
    get_f32(kv, "beta1", cfg.beta1);
    get_f32(kv, "beta2", cfg.beta2);
    get_f32(kv, "eps", cfg.eps);
    get_f32(kv, "weight_decay", cfg.weight_decay);
    get_u32(kv, "log_every", cfg.log_every);
    get_u32(kv, "seed", cfg.seed);
    get_f32(kv, "init_std", cfg.init_std);
    get_f32(kv, "ln_eps", cfg.ln_eps);
    get_u32(kv, "dump_outputs", cfg.dump_outputs);
    return cfg;
}

void save_u32_tensor(const std::vector<uint32_t>& data, const ttnn::Shape& shape, const std::string& path) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open u32 file for writing: " + path);
    }
    uint32_t ndim = shape.rank();
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(uint32_t));
    for (uint32_t i = 0; i < ndim; ++i) {
        uint32_t dim = shape[i];
        file.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
    }
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(uint32_t));
}

struct Linear3DAcc {
    Tensor weight;  // [out_dim, in_dim]
    Tensor bias;    // [1, 1, out_dim]
    Tensor d_weight;
    Tensor d_bias;
    Tensor out;
    Tensor d_out;

    uint32_t in_dim;
    uint32_t out_dim;

    Linear3DAcc(uint32_t batch, uint32_t seq, uint32_t in_d, uint32_t out_d, float init, MeshDevice& dev)
        : weight(make_init_tensor(ttnn::Shape({out_d, in_d}), init, dev, InitKind::Randn)),
          bias(make_zeros(ttnn::Shape({1, 1, out_d}), dev)),
          d_weight(make_zeros(ttnn::Shape({out_d, in_d}), dev)),
          d_bias(make_zeros(ttnn::Shape({1, 1, out_d}), dev)),
          out(make_zeros(ttnn::Shape({batch, seq, out_d}), dev)),
          d_out(make_zeros(ttnn::Shape({batch, seq, out_d}), dev)),
          in_dim(in_d),
          out_dim(out_d) {}

    Value* forward(Graph& g, Value* x) {
        auto* w = g.leaf(&weight, &d_weight, true);
        auto* b = g.leaf(&bias, &d_bias, true);

        out = ttnn::add(matmul_fp32_acc(*x->data, weight, false, true), bias);

        auto* v = g.node(&out, &d_out);
        v->parents = {x, w, b};

        v->backward_fn = [x, w, b, this, v]() {
            if (!v->grad) return;
            const auto& dout = *v->grad;

            if (x->requires_grad) {
                x->accumulate_grad(matmul_fp32_acc(dout, weight, false, false));
            }
            if (w->requires_grad) {
                auto dw = matmul_fp32_acc(dout, *x->data, true, false);
                auto dw_sum = ttnn::sum(dw, 0, false);
                w->accumulate_grad(dw_sum);
            }
            if (b->requires_grad) {
                auto sum0 = ttnn::sum(dout, 0, true);
                b->accumulate_grad(ttnn::sum(sum0, 1, true));
            }
        };

        return v;
    }
};

struct LayerNormManual {
    Tensor gamma;
    Tensor beta;
    Tensor d_gamma;
    Tensor d_beta;

    Tensor out;
    Tensor d_out;
    Tensor x_norm;
    Tensor rstd;

    uint32_t dim;
    float eps;

    LayerNormManual(uint32_t batch, uint32_t seq, uint32_t d, float epsilon, MeshDevice& dev)
        : gamma(make_full(ttnn::Shape({1, 1, d}), 1.0f, dev)),
          beta(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          d_gamma(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          d_beta(make_zeros(ttnn::Shape({1, 1, d}), dev)),
          out(make_zeros(ttnn::Shape({batch, seq, d}), dev)),
          d_out(make_zeros(ttnn::Shape({batch, seq, d}), dev)),
          x_norm(make_zeros(ttnn::Shape({batch, seq, d}), dev)),
          rstd(make_zeros(ttnn::Shape({batch, seq, 1}), dev)),
          dim(d),
          eps(epsilon) {}

    Value* forward(Graph& g, Value* x) {
        auto* gam = g.leaf(&gamma, &d_gamma, true);
        auto* bet = g.leaf(&beta, &d_beta, true);

        auto mean = ttnn::mean(*x->data, -1, true, std::nullopt, get_fp32_acc_compute_config());
        auto centered = ttnn::subtract(*x->data, mean);
        auto var = ttnn::mean(ttnn::multiply(centered, centered), -1, true, std::nullopt,
                              get_fp32_acc_compute_config());
        rstd = ttnn::rsqrt(ttnn::add(var, eps), false);
        x_norm = ttnn::multiply(centered, rstd);
        out = ttnn::add(ttnn::multiply(gamma, x_norm), beta);

        auto* v = g.node(&out, &d_out);
        v->parents = {x, gam, bet};

        v->backward_fn = [x, gam, bet, this, v]() {
            if (!v->grad) return;
            const auto& dout = *v->grad;

            if (bet->requires_grad) {
                auto sum0 = ttnn::sum(dout, 0, true);
                bet->accumulate_grad(ttnn::sum(sum0, 1, true));
            }
            if (gam->requires_grad) {
                auto d_out_x_norm = ttnn::multiply(dout, x_norm);
                auto sum0_g = ttnn::sum(d_out_x_norm, 0, true);
                gam->accumulate_grad(ttnn::sum(sum0_g, 1, true));
            }
            if (x->requires_grad) {
                auto d_x_norm = ttnn::multiply(dout, gamma);
                auto mean_d_x_norm = ttnn::mean(d_x_norm, -1, true, std::nullopt,
                                                 get_fp32_acc_compute_config());
                auto d_x_norm_x_norm = ttnn::multiply(d_x_norm, x_norm);
                auto mean_d_x_norm_x_norm = ttnn::mean(d_x_norm_x_norm, -1, true, std::nullopt,
                                                       get_fp32_acc_compute_config());
                auto diff1 = ttnn::subtract(d_x_norm, mean_d_x_norm);
                auto term2 = ttnn::multiply(x_norm, mean_d_x_norm_x_norm);
                auto diff2 = ttnn::subtract(diff1, term2);
                x->accumulate_grad(ttnn::multiply(rstd, diff2));
            }
        };

        return v;
    }
};

Value* reshape_op(Graph& g, Value* x, const ttnn::Shape& shape, Tensor* out, Tensor* d_out) {
    *out = ttnn::reshape(*x->data, shape);
    auto* v = g.node(out, d_out);
    v->parents = {x};
    const auto orig_shape = x->data->logical_shape();
    v->backward_fn = [x, v, orig_shape]() {
        if (!v->grad) return;
        x->accumulate_grad(ttnn::reshape(*v->grad, orig_shape));
    };
    return v;
}

static ttnn::SmallVector<int64_t> inverse_permute(const ttnn::SmallVector<int64_t>& dims) {
    ttnn::SmallVector<int64_t> inv(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        inv[dims[i]] = static_cast<int64_t>(i);
    }
    return inv;
}

Value* permute_op(Graph& g, Value* x, const ttnn::SmallVector<int64_t>& dims, Tensor* out, Tensor* d_out) {
    *out = ttnn::permute(*x->data, dims, std::nullopt, std::nullopt);
    auto* v = g.node(out, d_out);
    v->parents = {x};
    auto inv = inverse_permute(dims);
    v->backward_fn = [x, v, inv]() {
        if (!v->grad) return;
        x->accumulate_grad(ttnn::permute(*v->grad, inv, std::nullopt, std::nullopt));
    };
    return v;
}

Value* scale_op(Graph& g, Value* x, float scale, Tensor* out, Tensor* d_out) {
    *out = ttnn::multiply(*x->data, scale);
    auto* v = g.node(out, d_out);
    v->parents = {x};
    v->backward_fn = [x, v, scale]() {
        if (!v->grad) return;
        x->accumulate_grad(ttnn::multiply(*v->grad, scale));
    };
    return v;
}

struct GPT2Reimpl {
    Embedding tok;
    Embedding pos;
    LayerNormManual ln1;
    LayerNormManual ln2;

    Linear3DAcc wq;
    Linear3DAcc wk;
    Linear3DAcc wv;
    Linear3DAcc wo;

    Linear3DAcc ffn_w1;
    Linear3DAcc ffn_w2;

    Linear3DAcc lm_head;

    uint32_t batch;
    uint32_t seq;
    uint32_t dim;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t vocab;
    float scale;

    Tensor tok_plus_pos;
    Tensor d_tok_plus_pos;

    Tensor q;
    Tensor d_q;
    Tensor k;
    Tensor d_k;
    Tensor v;
    Tensor d_v;

    Tensor q_4d;
    Tensor d_q_4d;
    Tensor k_4d;
    Tensor d_k_4d;
    Tensor v_4d;
    Tensor d_v_4d;

    Tensor q_perm;
    Tensor d_q_perm;
    Tensor k_perm;
    Tensor d_k_perm;
    Tensor v_perm;
    Tensor d_v_perm;

    Tensor q_heads;
    Tensor d_q_heads;
    Tensor k_heads;
    Tensor d_k_heads;
    Tensor v_heads;
    Tensor d_v_heads;

    Tensor attn_scores;
    Tensor d_attn_scores;
    Tensor attn_scores_scaled;
    Tensor d_attn_scores_scaled;
    Tensor attn_scores_masked;
    Tensor d_attn_scores_masked;
    Tensor attn_weights;
    Tensor d_attn_weights;

    Tensor attn_out_heads;
    Tensor d_attn_out_heads;

    Tensor attn_out_4d;
    Tensor d_attn_out_4d;
    Tensor attn_out_perm;
    Tensor d_attn_out_perm;
    Tensor attn_out;
    Tensor d_attn_out;

    Tensor attn_proj;
    Tensor d_attn_proj;
    Tensor residual1;
    Tensor d_residual1;

    Tensor ffn_w1_out;
    Tensor d_ffn_w1_out;
    Tensor gelu_out;
    Tensor d_gelu_out;
    Tensor ffn_w2_out;
    Tensor d_ffn_w2_out;

    Tensor output;
    Tensor d_output;

    Tensor logits;

    Tensor causal_mask;

    GPT2Reimpl(uint32_t b, uint32_t s, uint32_t d, uint32_t h, uint32_t ffn_mult, uint32_t v,
               float init, float ln_eps, MeshDevice& dev)
        : tok(v, d, b, s, init, dev),
          pos(s, d, b, s, init, dev),
          ln1(b, s, d, ln_eps, dev),
          ln2(b, s, d, ln_eps, dev),
          wq(b, s, d, d, init, dev),
          wk(b, s, d, d, init, dev),
          wv(b, s, d, d, init, dev),
          wo(b, s, d, d, init, dev),
          ffn_w1(b, s, d, d * ffn_mult, init, dev),
          ffn_w2(b, s, d * ffn_mult, d, init, dev),
          lm_head(b, s, d, v, init, dev),
          batch(b),
          seq(s),
          dim(d),
          heads(h),
          head_dim(d / h),
          vocab(v),
          scale(1.0f / std::sqrt(static_cast<float>(d / h))),
          tok_plus_pos(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_tok_plus_pos(make_zeros(ttnn::Shape({b, s, d}), dev)),
          q(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_q(make_zeros(ttnn::Shape({b, s, d}), dev)),
          k(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_k(make_zeros(ttnn::Shape({b, s, d}), dev)),
          v(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_v(make_zeros(ttnn::Shape({b, s, d}), dev)),
          q_4d(make_zeros(ttnn::Shape({b, s, h, head_dim}), dev)),
          d_q_4d(make_zeros(ttnn::Shape({b, s, h, head_dim}), dev)),
          k_4d(make_zeros(ttnn::Shape({b, s, h, head_dim}), dev)),
          d_k_4d(make_zeros(ttnn::Shape({b, s, h, head_dim}), dev)),
          v_4d(make_zeros(ttnn::Shape({b, s, h, head_dim}), dev)),
          d_v_4d(make_zeros(ttnn::Shape({b, s, h, head_dim}), dev)),
          q_perm(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          d_q_perm(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          k_perm(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          d_k_perm(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          v_perm(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          d_v_perm(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          q_heads(make_zeros(ttnn::Shape({b * h, s, head_dim}), dev)),
          d_q_heads(make_zeros(ttnn::Shape({b * h, s, head_dim}), dev)),
          k_heads(make_zeros(ttnn::Shape({b * h, s, head_dim}), dev)),
          d_k_heads(make_zeros(ttnn::Shape({b * h, s, head_dim}), dev)),
          v_heads(make_zeros(ttnn::Shape({b * h, s, head_dim}), dev)),
          d_v_heads(make_zeros(ttnn::Shape({b * h, s, head_dim}), dev)),
          attn_scores(make_zeros(ttnn::Shape({b * h, s, s}), dev)),
          d_attn_scores(make_zeros(ttnn::Shape({b * h, s, s}), dev)),
          attn_scores_scaled(make_zeros(ttnn::Shape({b * h, s, s}), dev)),
          d_attn_scores_scaled(make_zeros(ttnn::Shape({b * h, s, s}), dev)),
          attn_scores_masked(make_zeros(ttnn::Shape({b * h, s, s}), dev)),
          d_attn_scores_masked(make_zeros(ttnn::Shape({b * h, s, s}), dev)),
          attn_weights(make_zeros(ttnn::Shape({b * h, s, s}), dev)),
          d_attn_weights(make_zeros(ttnn::Shape({b * h, s, s}), dev)),
          attn_out_heads(make_zeros(ttnn::Shape({b * h, s, head_dim}), dev)),
          d_attn_out_heads(make_zeros(ttnn::Shape({b * h, s, head_dim}), dev)),
          attn_out_4d(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          d_attn_out_4d(make_zeros(ttnn::Shape({b, h, s, head_dim}), dev)),
          attn_out_perm(make_zeros(ttnn::Shape({b, s, h, head_dim}), dev)),
          d_attn_out_perm(make_zeros(ttnn::Shape({b, s, h, head_dim}), dev)),
          attn_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_attn_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          attn_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_attn_proj(make_zeros(ttnn::Shape({b, s, d}), dev)),
          residual1(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_residual1(make_zeros(ttnn::Shape({b, s, d}), dev)),
          ffn_w1_out(make_zeros(ttnn::Shape({b, s, d * ffn_mult}), dev)),
          d_ffn_w1_out(make_zeros(ttnn::Shape({b, s, d * ffn_mult}), dev)),
          gelu_out(make_zeros(ttnn::Shape({b, s, d * ffn_mult}), dev)),
          d_gelu_out(make_zeros(ttnn::Shape({b, s, d * ffn_mult}), dev)),
          ffn_w2_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_ffn_w2_out(make_zeros(ttnn::Shape({b, s, d}), dev)),
          output(make_zeros(ttnn::Shape({b, s, d}), dev)),
          d_output(make_zeros(ttnn::Shape({b, s, d}), dev)),
          logits(make_zeros(ttnn::Shape({b, s, v}), dev)),
          causal_mask(create_causal_mask(s, dev)) {}

    static Tensor create_causal_mask(uint32_t s, MeshDevice& dev) {
        auto mask = make_full(ttnn::Shape({1, s, s}), -1e9f, dev);
        return ttnn::triu(mask, 1);
    }

    Value* forward(Graph& g, Value* tok_idx, Value* pos_idx) {
        auto* tok_v = tok.forward(g, tok_idx);
        auto* pos_v = pos.forward(g, pos_idx);
        auto* x = add(g, tok_v, pos_v, &tok_plus_pos, &d_tok_plus_pos);

        auto* ln1_v = ln1.forward(g, x);

        auto* q_v = wq.forward(g, ln1_v);
        q = *q_v->data;
        auto* k_v = wk.forward(g, ln1_v);
        k = *k_v->data;
        auto* v_v = wv.forward(g, ln1_v);
        v = *v_v->data;

        auto* q_4d_v = reshape_op(g, q_v, ttnn::Shape({batch, seq, heads, head_dim}), &q_4d, &d_q_4d);
        auto* k_4d_v = reshape_op(g, k_v, ttnn::Shape({batch, seq, heads, head_dim}), &k_4d, &d_k_4d);
        auto* v_4d_v = reshape_op(g, v_v, ttnn::Shape({batch, seq, heads, head_dim}), &v_4d, &d_v_4d);

        auto perm_dims = ttnn::SmallVector<int64_t>{0, 2, 1, 3};
        auto* q_perm_v = permute_op(g, q_4d_v, perm_dims, &q_perm, &d_q_perm);
        auto* k_perm_v = permute_op(g, k_4d_v, perm_dims, &k_perm, &d_k_perm);
        auto* v_perm_v = permute_op(g, v_4d_v, perm_dims, &v_perm, &d_v_perm);

        auto* q_heads_v = reshape_op(g, q_perm_v, ttnn::Shape({batch * heads, seq, head_dim}), &q_heads, &d_q_heads);
        auto* k_heads_v = reshape_op(g, k_perm_v, ttnn::Shape({batch * heads, seq, head_dim}), &k_heads, &d_k_heads);
        auto* v_heads_v = reshape_op(g, v_perm_v, ttnn::Shape({batch * heads, seq, head_dim}), &v_heads, &d_v_heads);

        attn_scores = matmul_fp32_acc(*q_heads_v->data, *k_heads_v->data, false, true);
        auto* scores_v = g.node(&attn_scores, &d_attn_scores);
        scores_v->parents = {q_heads_v, k_heads_v};
        scores_v->backward_fn = [q_heads_v, k_heads_v, scores_v]() {
            if (!scores_v->grad) return;
            const auto& dout = *scores_v->grad;
            if (q_heads_v->requires_grad) {
                q_heads_v->accumulate_grad(matmul_fp32_acc(dout, *k_heads_v->data));
            }
            if (k_heads_v->requires_grad) {
                k_heads_v->accumulate_grad(matmul_fp32_acc(dout, *q_heads_v->data, true, false));
            }
        };

        auto* scores_scaled_v = scale_op(g, scores_v, scale, &attn_scores_scaled, &d_attn_scores_scaled);

        auto* mask_v = g.leaf(&causal_mask, nullptr, false);
        auto* scores_masked_v = add(g, scores_scaled_v, mask_v, &attn_scores_masked, &d_attn_scores_masked);

        auto* attn_w_v = softmax(g, scores_masked_v, -1, &attn_weights, &d_attn_weights);

        attn_out_heads = matmul_fp32_acc(*attn_w_v->data, *v_heads_v->data, false, false);
        auto* attn_out_heads_v = g.node(&attn_out_heads, &d_attn_out_heads);
        attn_out_heads_v->parents = {attn_w_v, v_heads_v};
        attn_out_heads_v->backward_fn = [attn_w_v, v_heads_v, attn_out_heads_v]() {
            if (!attn_out_heads_v->grad) return;
            const auto& dout = *attn_out_heads_v->grad;
            if (attn_w_v->requires_grad) {
                attn_w_v->accumulate_grad(matmul_fp32_acc(dout, *v_heads_v->data, false, true));
            }
            if (v_heads_v->requires_grad) {
                v_heads_v->accumulate_grad(matmul_fp32_acc(*attn_w_v->data, dout, true, false));
            }
        };

        auto* attn_out_4d_v = reshape_op(g, attn_out_heads_v, ttnn::Shape({batch, heads, seq, head_dim}), &attn_out_4d, &d_attn_out_4d);
        auto perm_back = ttnn::SmallVector<int64_t>{0, 2, 1, 3};
        auto* attn_out_perm_v = permute_op(g, attn_out_4d_v, perm_back, &attn_out_perm, &d_attn_out_perm);
        auto* attn_out_v = reshape_op(g, attn_out_perm_v, ttnn::Shape({batch, seq, dim}), &attn_out, &d_attn_out);

        auto* attn_proj_v = wo.forward(g, attn_out_v);
        attn_proj = *attn_proj_v->data;

        auto* res1_v = add(g, x, attn_proj_v, &residual1, &d_residual1);

        auto* ln2_v = ln2.forward(g, res1_v);

        auto* ffn_w1_v = ffn_w1.forward(g, ln2_v);
        ffn_w1_out = *ffn_w1_v->data;
        auto* gelu_v = gelu(g, ffn_w1_v, &gelu_out, &d_gelu_out);
        auto* ffn_w2_v = ffn_w2.forward(g, gelu_v);
        ffn_w2_out = *ffn_w2_v->data;

        auto* out_v = add(g, res1_v, ffn_w2_v, &output, &d_output);

        auto* logits_v = lm_head.forward(g, out_v);
        logits = *logits_v->data;

        return logits_v;
    }
};

void write_meta(const std::filesystem::path& dir, const Config& cfg) {
    std::filesystem::create_directories(dir);
    std::ofstream file(dir / "meta.json");
    if (!file) return;
    file << "{\n";
    file << "  \"batch\": " << cfg.batch << ",\n";
    file << "  \"seq\": " << cfg.seq << ",\n";
    file << "  \"dim\": " << cfg.dim << ",\n";
    file << "  \"heads\": " << cfg.heads << ",\n";
    file << "  \"ffn_mult\": " << cfg.ffn_mult << ",\n";
    file << "  \"vocab\": " << cfg.vocab << ",\n";
    file << "  \"steps\": " << cfg.steps << ",\n";
    file << "  \"lr\": " << cfg.lr << ",\n";
    file << "  \"beta1\": " << cfg.beta1 << ",\n";
    file << "  \"beta2\": " << cfg.beta2 << ",\n";
    file << "  \"eps\": " << cfg.eps << ",\n";
    file << "  \"weight_decay\": " << cfg.weight_decay << ",\n";
    file << "  \"log_every\": " << cfg.log_every << ",\n";
    file << "  \"seed\": " << cfg.seed << ",\n";
    file << "  \"init_std\": " << cfg.init_std << ",\n";
    file << "  \"ln_eps\": " << cfg.ln_eps << "\n";
    file << "}\n";
}

} // namespace

int main() {
    const char* env_path = std::getenv("GPT2_REIMPL_CONFIG");
    std::string config_path = env_path
        ? env_path
        : "/home/howard/ttnn-perf/experiments/gpt2_reimpl/default.yaml";

    Config cfg = load_config(config_path);

    fmt::print("# GPT2 reimpl config: {}\n", config_path);
    fmt::print("# batch={}, seq={}, dim={}, heads={}, ffn_mult={}, vocab={}\n",
               cfg.batch, cfg.seq, cfg.dim, cfg.heads, cfg.ffn_mult, cfg.vocab);
    fmt::print("# steps={}, lr={}, beta1={}, beta2={}, eps={}, weight_decay={}\n",
               cfg.steps, cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);

    if (cfg.batch % 32 != 0 || cfg.seq % 32 != 0 || cfg.dim % 32 != 0 || cfg.vocab % 32 != 0) {
        fmt::print("ERROR: batch/seq/dim/vocab must be multiples of 32\n");
        return 1;
    }
    if (cfg.dim % cfg.heads != 0) {
        fmt::print("ERROR: dim must be divisible by heads\n");
        return 1;
    }

    DeviceGuard guard;
    MeshDevice& device = guard.get();

    g_seed_counter = std::max<uint32_t>(1, cfg.seed);

    // Build token + pos indices
    std::mt19937 gen(cfg.seed);
    std::uniform_int_distribution<uint32_t> dist(0, cfg.vocab - 1);

    std::vector<uint32_t> tok_indices(cfg.batch * cfg.seq);
    for (uint32_t i = 0; i < cfg.batch * cfg.seq; ++i) {
        tok_indices[i] = dist(gen);
    }

    std::vector<uint32_t> pos_indices(cfg.batch * cfg.seq);
    for (uint32_t b = 0; b < cfg.batch; ++b) {
        for (uint32_t s = 0; s < cfg.seq; ++s) {
            pos_indices[b * cfg.seq + s] = s;
        }
    }

    auto tok_idx_tensor = make_indices(tok_indices, ttnn::Shape({cfg.batch, cfg.seq}), device);
    auto pos_idx_tensor = make_indices(pos_indices, ttnn::Shape({cfg.batch, cfg.seq}), device);

    GPT2Reimpl model(cfg.batch, cfg.seq, cfg.dim, cfg.heads, cfg.ffn_mult, cfg.vocab,
                     cfg.init_std, cfg.ln_eps, device);

    Tensor target = make_randn(ttnn::Shape({cfg.batch, cfg.seq, cfg.vocab}), cfg.init_std, device);
    Tensor loss = make_zeros(ttnn::Shape({1, 1}), device);
    Tensor d_loss = make_zeros(ttnn::Shape({1, 1}), device);
    Tensor diff = make_zeros(ttnn::Shape({cfg.batch, cfg.seq, cfg.vocab}), device);

    Adam adam(cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);
    ADAM_REGISTER_EMBEDDING(adam, model.tok, device);
    ADAM_REGISTER_EMBEDDING(adam, model.pos, device);
    ADAM_REGISTER_LAYERNORM(adam, model.ln1, device);
    ADAM_REGISTER_LAYERNORM(adam, model.ln2, device);
    ADAM_REGISTER_LINEAR3D(adam, model.wq, device);
    ADAM_REGISTER_LINEAR3D(adam, model.wk, device);
    ADAM_REGISTER_LINEAR3D(adam, model.wv, device);
    ADAM_REGISTER_LINEAR3D(adam, model.wo, device);
    ADAM_REGISTER_LINEAR3D(adam, model.ffn_w1, device);
    ADAM_REGISTER_LINEAR3D(adam, model.ffn_w2, device);
    ADAM_REGISTER_LINEAR3D(adam, model.lm_head, device);

    auto outputs_dir = std::filesystem::path(cfg.outputs_dir) / "cpp";
    write_meta(outputs_dir, cfg);

    auto losses_path = outputs_dir / "losses.tsv";
    {
        std::ofstream file(losses_path);
        file << "step\tloss\n";
    }

    for (uint32_t step = 0; step < cfg.steps; ++step) {
        Graph graph;
        auto* tok_idx_v = graph.leaf(&tok_idx_tensor, nullptr, false);
        auto* pos_idx_v = graph.leaf(&pos_idx_tensor, nullptr, false);
        auto* logits_v = model.forward(graph, tok_idx_v, pos_idx_v);
        auto* loss_v = mse(graph, logits_v, &target, &loss, &d_loss, &diff);
        graph.build_topo(loss_v);

        adam.zero_grad();
        graph.zero_grad();
        graph.backward(loss_v);
        tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

        float loss_val = static_cast<float>(loss.cpu().to_vector<bfloat16>()[0]);
        {
            std::ofstream file(losses_path, std::ios::app);
            file << step << "\t" << loss_val << "\n";
        }

        if (cfg.dump_outputs != 0 && step == 0) {
            auto out_dir = outputs_dir / "step_0000";
            save_u32_tensor(tok_indices, ttnn::Shape({cfg.batch, cfg.seq}), (out_dir / "tokens_u32.bin").string());
            save_u32_tensor(pos_indices, ttnn::Shape({cfg.batch, cfg.seq}), (out_dir / "pos_u32.bin").string());

            traced::save_tensor(model.tok.weight, (out_dir / "tok_weight.bin").string(), &device);
            traced::save_tensor(model.pos.weight, (out_dir / "pos_weight.bin").string(), &device);
            traced::save_tensor(model.ln1.gamma, (out_dir / "ln1_gamma.bin").string(), &device);
            traced::save_tensor(model.ln1.beta, (out_dir / "ln1_beta.bin").string(), &device);
            traced::save_tensor(model.ln2.gamma, (out_dir / "ln2_gamma.bin").string(), &device);
            traced::save_tensor(model.ln2.beta, (out_dir / "ln2_beta.bin").string(), &device);

            traced::save_tensor(model.wq.weight, (out_dir / "wq_weight.bin").string(), &device);
            traced::save_tensor(model.wq.bias, (out_dir / "wq_bias.bin").string(), &device);
            traced::save_tensor(model.wk.weight, (out_dir / "wk_weight.bin").string(), &device);
            traced::save_tensor(model.wk.bias, (out_dir / "wk_bias.bin").string(), &device);
            traced::save_tensor(model.wv.weight, (out_dir / "wv_weight.bin").string(), &device);
            traced::save_tensor(model.wv.bias, (out_dir / "wv_bias.bin").string(), &device);
            traced::save_tensor(model.wo.weight, (out_dir / "wo_weight.bin").string(), &device);
            traced::save_tensor(model.wo.bias, (out_dir / "wo_bias.bin").string(), &device);

            traced::save_tensor(model.ffn_w1.weight, (out_dir / "ffn_w1_weight.bin").string(), &device);
            traced::save_tensor(model.ffn_w1.bias, (out_dir / "ffn_w1_bias.bin").string(), &device);
            traced::save_tensor(model.ffn_w2.weight, (out_dir / "ffn_w2_weight.bin").string(), &device);
            traced::save_tensor(model.ffn_w2.bias, (out_dir / "ffn_w2_bias.bin").string(), &device);

            traced::save_tensor(model.lm_head.weight, (out_dir / "lm_head_weight.bin").string(), &device);
            traced::save_tensor(model.lm_head.bias, (out_dir / "lm_head_bias.bin").string(), &device);

            traced::save_tensor(model.tok.out, (out_dir / "tok_embed.bin").string(), &device);
            traced::save_tensor(model.pos.out, (out_dir / "pos_embed.bin").string(), &device);
            traced::save_tensor(model.tok_plus_pos, (out_dir / "tok_plus_pos.bin").string(), &device);
            traced::save_tensor(model.ln1.out, (out_dir / "ln1_out.bin").string(), &device);

            traced::save_tensor(model.q, (out_dir / "q.bin").string(), &device);
            traced::save_tensor(model.k, (out_dir / "k.bin").string(), &device);
            traced::save_tensor(model.v, (out_dir / "v.bin").string(), &device);
            traced::save_tensor(model.q_heads, (out_dir / "q_heads.bin").string(), &device);
            traced::save_tensor(model.k_heads, (out_dir / "k_heads.bin").string(), &device);
            traced::save_tensor(model.v_heads, (out_dir / "v_heads.bin").string(), &device);
            traced::save_tensor(model.attn_scores_scaled, (out_dir / "attn_scores.bin").string(), &device);
            traced::save_tensor(model.attn_scores_masked, (out_dir / "attn_scores_masked.bin").string(), &device);
            traced::save_tensor(model.attn_weights, (out_dir / "attn_weights.bin").string(), &device);
            traced::save_tensor(model.attn_out, (out_dir / "attn_out.bin").string(), &device);
            traced::save_tensor(model.attn_proj, (out_dir / "attn_proj.bin").string(), &device);
            traced::save_tensor(model.residual1, (out_dir / "residual1.bin").string(), &device);
            traced::save_tensor(model.ln2.out, (out_dir / "ln2_out.bin").string(), &device);
            traced::save_tensor(model.ffn_w1_out, (out_dir / "ffn_w1_out.bin").string(), &device);
            traced::save_tensor(model.gelu_out, (out_dir / "ffn_gelu.bin").string(), &device);
            traced::save_tensor(model.ffn_w2_out, (out_dir / "ffn_w2_out.bin").string(), &device);
            traced::save_tensor(model.output, (out_dir / "output.bin").string(), &device);

            traced::save_tensor(model.logits, (out_dir / "logits.bin").string(), &device);
            traced::save_tensor(loss, (out_dir / "loss.bin").string(), &device);
            traced::save_tensor(target, (out_dir / "target.bin").string(), &device);

            traced::save_tensor(model.tok.d_weight, (out_dir / "tok_weight_grad.bin").string(), &device);
            traced::save_tensor(model.pos.d_weight, (out_dir / "pos_weight_grad.bin").string(), &device);
            traced::save_tensor(model.ln1.d_gamma, (out_dir / "ln1_gamma_grad.bin").string(), &device);
            traced::save_tensor(model.ln1.d_beta, (out_dir / "ln1_beta_grad.bin").string(), &device);
            traced::save_tensor(model.ln2.d_gamma, (out_dir / "ln2_gamma_grad.bin").string(), &device);
            traced::save_tensor(model.ln2.d_beta, (out_dir / "ln2_beta_grad.bin").string(), &device);

            traced::save_tensor(model.wq.d_weight, (out_dir / "wq_weight_grad.bin").string(), &device);
            traced::save_tensor(model.wq.d_bias, (out_dir / "wq_bias_grad.bin").string(), &device);
            traced::save_tensor(model.wk.d_weight, (out_dir / "wk_weight_grad.bin").string(), &device);
            traced::save_tensor(model.wk.d_bias, (out_dir / "wk_bias_grad.bin").string(), &device);
            traced::save_tensor(model.wv.d_weight, (out_dir / "wv_weight_grad.bin").string(), &device);
            traced::save_tensor(model.wv.d_bias, (out_dir / "wv_bias_grad.bin").string(), &device);
            traced::save_tensor(model.wo.d_weight, (out_dir / "wo_weight_grad.bin").string(), &device);
            traced::save_tensor(model.wo.d_bias, (out_dir / "wo_bias_grad.bin").string(), &device);

            traced::save_tensor(model.ffn_w1.d_weight, (out_dir / "ffn_w1_weight_grad.bin").string(), &device);
            traced::save_tensor(model.ffn_w1.d_bias, (out_dir / "ffn_w1_bias_grad.bin").string(), &device);
            traced::save_tensor(model.ffn_w2.d_weight, (out_dir / "ffn_w2_weight_grad.bin").string(), &device);
            traced::save_tensor(model.ffn_w2.d_bias, (out_dir / "ffn_w2_bias_grad.bin").string(), &device);

            traced::save_tensor(model.lm_head.d_weight, (out_dir / "lm_head_weight_grad.bin").string(), &device);
            traced::save_tensor(model.lm_head.d_bias, (out_dir / "lm_head_bias_grad.bin").string(), &device);

            traced::save_tensor(model.lm_head.d_out, (out_dir / "logits_grad.bin").string(), &device);
        }

        if (cfg.log_every != 0 && (step % cfg.log_every == 0 || step + 1 == cfg.steps)) {
            fmt::print("step {} | loss {:.6f}\n", step, loss_val);
        }

        adam.step();
    }

    fmt::print("# Finished {} steps\n", cfg.steps);
    return 0;
}
