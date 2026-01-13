// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Load Transformer layer inputs/params from TDF1, run TTNN layer forward, save outputs.

#include "tensordiff/core/serialize.hpp"
#include "tensordiff/core/ttnn_bridge.hpp"

#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/eltwise/unary/unary_composite.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
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

static CpuTensor squeeze_batch(const CpuTensor& t) {
    if (t.shape().size() != 3 || t.shape()[0] != 1) {
        return t;
    }
    uint32_t s = t.shape()[1];
    uint32_t d = t.shape()[2];
    CpuTensor out({s, d}, t.dtype());
    const size_t bytes = static_cast<size_t>(s) * d * tensordiff::core::dtype_size(t.dtype());
    std::memcpy(out.raw().data(), t.raw().data(), bytes);
    return out;
}

static ttnn::WormholeComputeKernelConfig get_compute_cfg() {
    return ttnn::WormholeComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi2,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = true,
    };
}

static tt::tt_metal::Tensor matmul_fp32_acc(const tt::tt_metal::Tensor& a,
                                            const tt::tt_metal::Tensor& b,
                                            bool transpose_a = false,
                                            bool transpose_b = false) {
    return ttnn::matmul(
        a,
        b,
        transpose_a,
        transpose_b,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        get_compute_cfg());
}

static tt::tt_metal::Tensor split_heads(const tt::tt_metal::Tensor& x,
                                        uint32_t batch, uint32_t seq,
                                        uint32_t heads, uint32_t head_dim) {
    auto reshaped = ttnn::reshape(x, ttnn::Shape({batch, seq, heads, head_dim}));
    ttnn::SmallVector<int64_t> dims = {0, 2, 1, 3};
    return ttnn::permute(reshaped, dims);
}

static tt::tt_metal::Tensor merge_heads(const tt::tt_metal::Tensor& x,
                                        uint32_t batch, uint32_t seq,
                                        uint32_t heads, uint32_t head_dim) {
    ttnn::SmallVector<int64_t> dims = {0, 2, 1, 3};
    auto permuted = ttnn::permute(x, dims);
    return ttnn::reshape(permuted, ttnn::Shape({batch, seq, heads * head_dim}));
}

static tt::tt_metal::Tensor gelu_forward(const tt::tt_metal::Tensor& x) {
    bool fast = false;
    const char* env = std::getenv("GELU_APPROX");
    if (env && std::string(env) == "tanh") {
        fast = true;
    }
    return ttnn::gelu(x, fast);
}

int main() {
    DeviceGuard guard;
    auto& device = guard.get();

    const char* out_env = std::getenv("TENSORDIFF_OUT_DIR");
    std::filesystem::path out_dir = out_env ? out_env : "tensordiff/outputs/transformer_layer_torch_rand";

    uint32_t B = 0, S = 0, D = 0, H = 1, FFN_MULT = 4;
    if (const char* env = std::getenv("TDF_H")) H = static_cast<uint32_t>(std::stoul(env));
    if (const char* env = std::getenv("TDF_FFN_MULT")) FFN_MULT = static_cast<uint32_t>(std::stoul(env));

    float eps = 1e-5f;
    if (const char* env = std::getenv("TDF_EPS")) eps = std::stof(env);

    // Load input + params
    auto x = load_tensor((out_dir / "torch_tx_in.bin").string());
    if (x.shape().size() != 3) {
        throw std::runtime_error("x shape must be [B,S,D]");
    }
    B = x.shape()[0];
    S = x.shape()[1];
    D = x.shape()[2];

    if (const char* env = std::getenv("TDF_B")) {
        uint32_t env_b = static_cast<uint32_t>(std::stoul(env));
        if (env_b != B) throw std::runtime_error("TDF_B mismatch with x shape");
    }
    if (const char* env = std::getenv("TDF_S")) {
        uint32_t env_s = static_cast<uint32_t>(std::stoul(env));
        if (env_s != S) throw std::runtime_error("TDF_S mismatch with x shape");
    }
    if (const char* env = std::getenv("TDF_D")) {
        uint32_t env_d = static_cast<uint32_t>(std::stoul(env));
        if (env_d != D) throw std::runtime_error("TDF_D mismatch with x shape");
    }

    const uint32_t ffn_dim = D * FFN_MULT;
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

    if (w1.shape().size() != 2 || w1.shape()[0] != ffn_dim || w1.shape()[1] != D) {
        throw std::runtime_error("w1 shape mismatch for ffn_dim");
    }
    if (w2.shape().size() != 2 || w2.shape()[0] != D || w2.shape()[1] != ffn_dim) {
        throw std::runtime_error("w2 shape mismatch for ffn_dim");
    }

    // TTNN tensors
    auto x_ttnn = tensordiff::core::to_ttnn(x, device, false);
    auto ln1_gamma_t = tensordiff::core::to_ttnn(ln1_gamma, device, false);
    auto ln1_beta_t  = tensordiff::core::to_ttnn(ln1_beta, device, false);
    auto ln2_gamma_t = tensordiff::core::to_ttnn(ln2_gamma, device, false);
    auto ln2_beta_t  = tensordiff::core::to_ttnn(ln2_beta, device, false);

    auto wq_t = tensordiff::core::to_ttnn(wq, device, false);
    auto wk_t = tensordiff::core::to_ttnn(wk, device, false);
    auto wv_t = tensordiff::core::to_ttnn(wv, device, false);
    auto wo_t = tensordiff::core::to_ttnn(wo, device, false);

    auto bq_t = tensordiff::core::to_ttnn(bq, device, false);
    auto bk_t = tensordiff::core::to_ttnn(bk, device, false);
    auto bv_t = tensordiff::core::to_ttnn(bv, device, false);
    auto bo_t = tensordiff::core::to_ttnn(bo, device, false);

    auto w1_t = tensordiff::core::to_ttnn(w1, device, false);
    auto b1_t = tensordiff::core::to_ttnn(b1, device, false);
    auto w2_t = tensordiff::core::to_ttnn(w2, device, false);
    auto b2_t = tensordiff::core::to_ttnn(b2, device, false);
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    // Forward: LN1
    auto ln1_out = ttnn::layer_norm(x_ttnn, eps, ln1_gamma_t, ln1_beta_t,
                                    std::nullopt, std::nullopt, std::nullopt, get_compute_cfg());

    // QKV
    auto q = ttnn::add(ttnn::matmul(ln1_out, wq_t, false, true), bq_t);
    auto k = ttnn::add(ttnn::matmul(ln1_out, wk_t, false, true), bk_t);
    auto v = ttnn::add(ttnn::matmul(ln1_out, wv_t, false, true), bv_t);

    uint32_t head_dim = D / H;
    auto qh = split_heads(q, B, S, H, head_dim);
    auto kh = split_heads(k, B, S, H, head_dim);
    auto vh = split_heads(v, B, S, H, head_dim);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto scores = ttnn::multiply(matmul_fp32_acc(qh, kh, false, true), scale);

    auto mask = ttnn::triu(
        ttnn::full(ttnn::Shape({1, 1, S, S}), -1e9f, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, device),
        1);
    auto scores_masked = ttnn::add(scores, mask);
    auto attn = ttnn::softmax(scores_masked, -1, std::nullopt, get_compute_cfg(), true);

    auto attn_out_h = matmul_fp32_acc(attn, vh);
    auto attn_out = merge_heads(attn_out_h, B, S, H, head_dim);
    auto attn_proj = ttnn::add(ttnn::matmul(attn_out, wo_t, false, true), bo_t);

    auto residual1 = ttnn::add(x_ttnn, attn_proj);

    // LN2
    auto ln2_out = ttnn::layer_norm(residual1, eps, ln2_gamma_t, ln2_beta_t,
                                    std::nullopt, std::nullopt, std::nullopt, get_compute_cfg());

    // FFN
    auto ffn1 = ttnn::add(ttnn::matmul(ln2_out, w1_t, false, true), b1_t);
    auto ffn_gelu = gelu_forward(ffn1);
    auto ffn2 = ttnn::add(ttnn::matmul(ffn_gelu, w2_t, false, true), b2_t);

    auto out = ttnn::add(residual1, ffn2);
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    auto x_round = tensordiff::core::from_ttnn(x_ttnn);
    auto out_cpu = tensordiff::core::from_ttnn(out);

    auto x_2d = squeeze_batch(x_round);
    auto out_2d = squeeze_batch(out_cpu);

    save_tensor(x_round, (out_dir / "ttnn_tx_in.bin").string(), "ttnn");
    save_tensor(out_cpu, (out_dir / "ttnn_tx_out.bin").string(), "ttnn");
    save_tensor(x_2d, (out_dir / "ttnn_tx_in_2d.bin").string(), "ttnn");
    save_tensor(out_2d, (out_dir / "ttnn_tx_out_2d.bin").string(), "ttnn");

    fmt::print("# wrote: {}\n", out_dir.string());
    return 0;
}
