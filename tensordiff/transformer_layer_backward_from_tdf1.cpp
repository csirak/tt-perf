// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Load Transformer layer inputs/params + upstream grad from TDF1, run autograd backward, save grads.

#include "tensordiff/core/serialize.hpp"
#include "tensordiff/core/ttnn_bridge.hpp"

#include "autograd/nn/linear.hpp"
#include "autograd/nn/ffn.hpp"
#include "autograd/nn/layer_norm.hpp"
#include "autograd/nn/attention.hpp"
#include "autograd/ops.hpp"
#include "autograd/value.hpp"

#include <ttnn/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>

#include <cstdlib>
#include <filesystem>
#include <cstring>

using tensordiff::core::CpuTensor;
using tensordiff::core::load_tensor;
using tensordiff::core::save_tensor;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using DispatchCoreConfig = tt::tt_metal::DispatchCoreConfig;
using DispatchCoreType = tt::tt_metal::DispatchCoreType;

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

int main() {
    DeviceGuard guard;
    auto& device = guard.get();

    const char* out_env = std::getenv("TENSORDIFF_OUT_DIR");
    std::filesystem::path out_dir = out_env ? out_env : "tensordiff/outputs/transformer_layer_bwd_torch_rand";

    float eps = 1e-5f;
    if (const char* env = std::getenv("TDF_EPS")) eps = std::stof(env);

    uint32_t H = 1;
    if (const char* env = std::getenv("TDF_H")) H = static_cast<uint32_t>(std::stoul(env));
    uint32_t FFN_MULT = 4;
    if (const char* env = std::getenv("TDF_FFN_MULT")) FFN_MULT = static_cast<uint32_t>(std::stoul(env));

    // Load input + params
    auto x = load_tensor((out_dir / "torch_tx_in.bin").string());
    if (x.shape().size() != 3) {
        throw std::runtime_error("x shape must be [B,S,D]");
    }
    uint32_t B = x.shape()[0];
    uint32_t S = x.shape()[1];
    uint32_t D = x.shape()[2];

    auto d_out = load_tensor((out_dir / "torch_d_out.bin").string());

    auto ln1_gamma = load_tensor((out_dir / "torch_ln1_gamma.bin").string());
    auto ln1_beta = load_tensor((out_dir / "torch_ln1_beta.bin").string());
    auto ln2_gamma = load_tensor((out_dir / "torch_ln2_gamma.bin").string());
    auto ln2_beta = load_tensor((out_dir / "torch_ln2_beta.bin").string());

    auto wq = load_tensor((out_dir / "torch_wq_weight.bin").string());
    auto wk = load_tensor((out_dir / "torch_wk_weight.bin").string());
    auto wv = load_tensor((out_dir / "torch_wv_weight.bin").string());
    auto wo = load_tensor((out_dir / "torch_wo_weight.bin").string());

    auto bq = load_tensor((out_dir / "torch_wq_bias.bin").string());
    auto bk = load_tensor((out_dir / "torch_wk_bias.bin").string());
    auto bv = load_tensor((out_dir / "torch_wv_bias.bin").string());
    auto bo = load_tensor((out_dir / "torch_wo_bias.bin").string());

    auto w1 = load_tensor((out_dir / "torch_ffn_w1_weight.bin").string());
    auto b1 = load_tensor((out_dir / "torch_ffn_w1_bias.bin").string());
    auto w2 = load_tensor((out_dir / "torch_ffn_w2_weight.bin").string());
    auto b2 = load_tensor((out_dir / "torch_ffn_w2_bias.bin").string());

    if (w1.shape().size() != 2 || w1.shape()[0] != D * FFN_MULT || w1.shape()[1] != D) {
        throw std::runtime_error("w1 shape mismatch for ffn_dim");
    }
    if (w2.shape().size() != 2 || w2.shape()[0] != D || w2.shape()[1] != D * FFN_MULT) {
        throw std::runtime_error("w2 shape mismatch for ffn_dim");
    }

    static_autograd::Graph g;
    static_autograd::PersistentTransformerLayerLN layer(
        B, S, D, H, FFN_MULT, 0.02f, device, eps);

    // Override parameters
    layer.ln1.gamma = tensordiff::core::to_ttnn(ln1_gamma, device, false);
    layer.ln1.beta  = tensordiff::core::to_ttnn(ln1_beta, device, false);
    layer.ln2.gamma = tensordiff::core::to_ttnn(ln2_gamma, device, false);
    layer.ln2.beta  = tensordiff::core::to_ttnn(ln2_beta, device, false);

    layer.wq.weight = tensordiff::core::to_ttnn(wq, device, false);
    layer.wk.weight = tensordiff::core::to_ttnn(wk, device, false);
    layer.wv.weight = tensordiff::core::to_ttnn(wv, device, false);
    layer.wo.weight = tensordiff::core::to_ttnn(wo, device, false);

    layer.wq.bias = tensordiff::core::to_ttnn(bq, device, false);
    layer.wk.bias = tensordiff::core::to_ttnn(bk, device, false);
    layer.wv.bias = tensordiff::core::to_ttnn(bv, device, false);
    layer.wo.bias = tensordiff::core::to_ttnn(bo, device, false);

    layer.ffn.w1.weight = tensordiff::core::to_ttnn(w1, device, false);
    layer.ffn.w1.bias   = tensordiff::core::to_ttnn(b1, device, false);
    layer.ffn.w2.weight = tensordiff::core::to_ttnn(w2, device, false);
    layer.ffn.w2.bias   = tensordiff::core::to_ttnn(b2, device, false);

    // Input + upstream grad
    auto x_t = tensordiff::core::to_ttnn(x, device, false);
    auto d_x = static_autograd::make_zeros(ttnn::Shape({B, S, D}), device);
    auto* x_v = g.leaf(&x_t, &d_x, true);

    auto* out_v = layer.forward(g, x_v);

    auto d_out_t = tensordiff::core::to_ttnn(d_out, device, false);
    auto* d_out_v = g.leaf(&d_out_t, nullptr, false);

    auto prod = static_autograd::make_zeros(ttnn::Shape({B, S, D}), device);
    auto d_prod = static_autograd::make_zeros(ttnn::Shape({B, S, D}), device);
    auto* prod_v = static_autograd::mul(g, out_v, d_out_v, &prod, &d_prod);

    g.zero_grad();
    g.backward(prod_v);
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    // Save grads
    save_tensor(tensordiff::core::from_ttnn(d_x), (out_dir / "ttnn_d_x.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.d_output), (out_dir / "ttnn_d_output.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.d_residual1), (out_dir / "ttnn_d_residual1.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.wo.d_out), (out_dir / "ttnn_d_attn_proj.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.d_attn_out), (out_dir / "ttnn_d_attn_out.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.d_attn_weights), (out_dir / "ttnn_d_attn_weights.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.d_attn_scores), (out_dir / "ttnn_d_attn_scores.bin").string(), "ttnn");

    save_tensor(tensordiff::core::from_ttnn(layer.wq.d_out), (out_dir / "ttnn_d_q.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.wk.d_out), (out_dir / "ttnn_d_k.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.wv.d_out), (out_dir / "ttnn_d_v.bin").string(), "ttnn");

    save_tensor(tensordiff::core::from_ttnn(layer.ln1.d_out), (out_dir / "ttnn_d_ln1.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.ln2.d_out), (out_dir / "ttnn_d_ln2.bin").string(), "ttnn");

    save_tensor(tensordiff::core::from_ttnn(layer.ffn.w2.d_out), (out_dir / "ttnn_d_ffn2.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.ffn.d_gelu_out), (out_dir / "ttnn_d_ffn_gelu.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.ffn.w1.d_out), (out_dir / "ttnn_d_ffn1.bin").string(), "ttnn");

    // Parameter grads
    save_tensor(tensordiff::core::from_ttnn(layer.wq.d_weight), (out_dir / "ttnn_wq_weight_grad.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.wk.d_weight), (out_dir / "ttnn_wk_weight_grad.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.wv.d_weight), (out_dir / "ttnn_wv_weight_grad.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.wo.d_weight), (out_dir / "ttnn_wo_weight_grad.bin").string(), "ttnn");

    save_tensor(tensordiff::core::from_ttnn(layer.wq.d_bias), (out_dir / "ttnn_wq_bias_grad.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.wk.d_bias), (out_dir / "ttnn_wk_bias_grad.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.wv.d_bias), (out_dir / "ttnn_wv_bias_grad.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.wo.d_bias), (out_dir / "ttnn_wo_bias_grad.bin").string(), "ttnn");

    save_tensor(tensordiff::core::from_ttnn(layer.ffn.w1.d_weight), (out_dir / "ttnn_ffn_w1_weight_grad.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.ffn.w1.d_bias), (out_dir / "ttnn_ffn_w1_bias_grad.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.ffn.w2.d_weight), (out_dir / "ttnn_ffn_w2_weight_grad.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.ffn.w2.d_bias), (out_dir / "ttnn_ffn_w2_bias_grad.bin").string(), "ttnn");

    save_tensor(tensordiff::core::from_ttnn(layer.ln1.d_gamma), (out_dir / "ttnn_ln1_gamma_grad.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.ln1.d_beta), (out_dir / "ttnn_ln1_beta_grad.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.ln2.d_gamma), (out_dir / "ttnn_ln2_gamma_grad.bin").string(), "ttnn");
    save_tensor(tensordiff::core::from_ttnn(layer.ln2.d_beta), (out_dir / "ttnn_ln2_beta_grad.bin").string(), "ttnn");

    fmt::print("# wrote: {}\n", out_dir.string());
    return 0;
}
