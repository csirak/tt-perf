// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Attention backward verification (TTNN C++ vs PyTorch).

#include "autograd/nn.hpp"
#include "autograd/utils/config.hpp"
#include "autograd/utils/tensor_io.hpp"

#include <ttnn/device.hpp>
#include <ttnn/operations/data_movement/tilize/tilize.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
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
    uint32_t batch = 32;
    uint32_t seq = 32;
    uint32_t dim = 128;
    uint32_t heads = 4;
    uint32_t seed = 1;
    float init_std = 0.02f;
    std::string dump_dir = "/home/howard/ttnn-perf/experiments/grok/outputs/attn_bw";
};

Config load_config(const std::string& path) {
    Config cfg;
    auto kv = load_kv_file(path);
    if (kv.empty()) {
        fmt::print("# Config not found: {} (using defaults)\n", path);
        return cfg;
    }
    get_u32(kv, "batch", cfg.batch);
    get_u32(kv, "seq", cfg.seq);
    get_u32(kv, "dim", cfg.dim);
    get_u32(kv, "heads", cfg.heads);
    get_u32(kv, "seed", cfg.seed);
    get_f32(kv, "init_std", cfg.init_std);
    get_str(kv, "dump_dir", cfg.dump_dir);
    return cfg;
}

std::vector<float> make_random_vec(size_t n, uint32_t seed, float scale = 1.0f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(gen) * scale;
    }
    return data;
}

Tensor make_host_bf16(const std::vector<float>& data, const ttnn::Shape& shape) {
    std::vector<bfloat16> bf(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        bf[i] = bfloat16(data[i]);
    }
    auto layout = ttnn::TensorLayout(
        ttnn::DataType::BFLOAT16,
        ttnn::PageConfig(ttnn::ROW_MAJOR_LAYOUT),
        tt::tt_metal::MemoryConfig{}
    );
    return Tensor::from_vector(bf, ttnn::TensorSpec(shape, layout));
}

Tensor create_causal_mask(uint32_t s, MeshDevice& dev) {
    auto mask = make_full(ttnn::Shape({1, 1, s, s}), -1e9f, dev);
    return ttnn::triu(mask, 1);
}

void write_meta(const std::filesystem::path& dir, const Config& cfg, float scale) {
    std::filesystem::create_directories(dir);
    std::ofstream file(dir / "meta.json");
    if (!file) {
        return;
    }
    file << "{\n";
    file << "  \"batch\": " << cfg.batch << ",\n";
    file << "  \"seq\": " << cfg.seq << ",\n";
    file << "  \"dim\": " << cfg.dim << ",\n";
    file << "  \"heads\": " << cfg.heads << ",\n";
    file << "  \"seed\": " << cfg.seed << ",\n";
    file << "  \"init_std\": " << cfg.init_std << ",\n";
    file << "  \"scale\": " << scale << "\n";
    file << "}\n";
}

}  // namespace

int main() {
    const char* env_path = std::getenv("ATTN_BW_CONFIG");
    std::string config_path = env_path
        ? env_path
        : "/home/howard/ttnn-perf/experiments/grok/attn_bw_verify.yaml";

    Config cfg = load_config(config_path);
    fmt::print("# Attention BW config: {}\n", config_path);
    fmt::print("# batch={}, seq={}, dim={}, heads={}\n", cfg.batch, cfg.seq, cfg.dim, cfg.heads);

    if (cfg.batch % 32 != 0 || cfg.seq % 32 != 0 || cfg.dim % 32 != 0) {
        fmt::print("ERROR: batch, seq, dim must be multiples of 32\n");
        return 1;
    }
    if (cfg.dim % cfg.heads != 0) {
        fmt::print("ERROR: dim must be divisible by heads\n");
        return 1;
    }

    DeviceGuard guard;
    MeshDevice& device = guard.get();

    g_seed_counter = std::max<uint32_t>(1, cfg.seed);

    const size_t numel = static_cast<size_t>(cfg.batch) * cfg.seq * cfg.dim;
    auto q_data = make_random_vec(numel, cfg.seed, cfg.init_std);
    auto k_data = make_random_vec(numel, cfg.seed + 1, cfg.init_std);
    auto v_data = make_random_vec(numel, cfg.seed + 2, cfg.init_std);
    auto target_data = make_random_vec(numel, cfg.seed + 3, cfg.init_std);

    auto q_host = make_host_bf16(q_data, ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}));
    auto k_host = make_host_bf16(k_data, ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}));
    auto v_host = make_host_bf16(v_data, ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}));
    auto target_host = make_host_bf16(target_data, ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}));

    Tensor q = ttnn::tilize(q_host.to_device(&device));
    Tensor k = ttnn::tilize(k_host.to_device(&device));
    Tensor v = ttnn::tilize(v_host.to_device(&device));
    Tensor target = ttnn::tilize(target_host.to_device(&device));

    Tensor d_q = make_zeros(ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}), device);
    Tensor d_k = make_zeros(ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}), device);
    Tensor d_v = make_zeros(ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}), device);

    Tensor attn_scores = make_zeros(ttnn::Shape({cfg.batch, cfg.heads, cfg.seq, cfg.seq}), device);
    Tensor attn_weights = make_zeros(ttnn::Shape({cfg.batch, cfg.heads, cfg.seq, cfg.seq}), device);
    Tensor attn_out_heads = make_zeros(ttnn::Shape({cfg.batch, cfg.heads, cfg.seq, cfg.dim / cfg.heads}), device);
    Tensor attn_out = make_zeros(ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}), device);
    Tensor d_attn_out = make_zeros(ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}), device);

    uint32_t batch = cfg.batch;
    uint32_t seq = cfg.seq;
    uint32_t heads = cfg.heads;
    uint32_t head_dim = cfg.dim / cfg.heads;
    float scale = attn_scale(cfg.dim, cfg.heads);
    Tensor causal_mask = create_causal_mask(cfg.seq, device);

    Graph graph;
    auto* q_node = graph.leaf(&q, &d_q, true);
    auto* k_node = graph.leaf(&k, &d_k, true);
    auto* v_node = graph.leaf(&v, &d_v, true);

    // Forward attention (multi-head)
    auto q_heads = split_heads(*q_node->data, batch, seq, heads, head_dim);
    auto k_heads = split_heads(*k_node->data, batch, seq, heads, head_dim);
    auto v_heads = split_heads(*v_node->data, batch, seq, heads, head_dim);

    attn_scores = ttnn::multiply(matmul_fp32_acc(q_heads, k_heads, false, true), scale);
    auto scores_masked = ttnn::add(attn_scores, causal_mask);
    attn_weights = ttnn::softmax(scores_masked, -1, std::nullopt, get_softmax_compute_config(), true);
    attn_out_heads = matmul_fp32_acc(attn_weights, v_heads);
    attn_out = merge_heads(attn_out_heads, batch, seq, heads, head_dim);

    auto* attn_node = graph.node(&attn_out, &d_attn_out);
    attn_node->parents = {q_node, k_node, v_node};
    attn_node->backward_fn = [attn_node, q_node, k_node, v_node, &attn_weights, &q_heads, &k_heads, &v_heads, batch, seq, heads, head_dim, scale]() {
        if (!attn_node->grad) {
            return;
        }
        const auto& dout = *attn_node->grad;

        auto d_out_heads = split_heads(dout, batch, seq, heads, head_dim);

        // d_attn = d_out_heads @ V.T
        auto d_attn = matmul_fp32_acc(d_out_heads, v_heads, false, true);

        // d_V = attn_weights.T @ d_out_heads
        if (v_node->requires_grad) {
            auto d_v_heads = matmul_fp32_acc(attn_weights, d_out_heads, true, false);
            v_node->accumulate_grad(merge_heads(d_v_heads, batch, seq, heads, head_dim));
        }

        // Softmax backward
        auto dy_y = ttnn::multiply(d_attn, attn_weights);
        auto sum_dy_y = ttnn::sum(dy_y, -1, true, std::nullopt, get_fp32_acc_compute_config());
        auto d_scores = ttnn::multiply(attn_weights, ttnn::subtract(d_attn, sum_dy_y));
        auto d_scores_scaled = ttnn::multiply(d_scores, scale);

        // d_Q = d_scores_scaled @ K
        if (q_node->requires_grad) {
            auto d_q_heads = matmul_fp32_acc(d_scores_scaled, k_heads);
            q_node->accumulate_grad(merge_heads(d_q_heads, batch, seq, heads, head_dim));
        }
        // d_K = d_scores_scaled.T @ Q
        if (k_node->requires_grad) {
            auto d_k_heads = matmul_fp32_acc(d_scores_scaled, q_heads, true, false);
            k_node->accumulate_grad(merge_heads(d_k_heads, batch, seq, heads, head_dim));
        }
    };

    MeanSquaredError mse(cfg.batch, cfg.seq, cfg.dim, device);
    auto* target_node = graph.leaf(&target, nullptr, false);
    auto* loss_node = mse.build(graph, attn_node, target_node);
    graph.build_topo(loss_node);

    mse.execute_forward(attn_out, target);
    graph.zero_grad();
    graph.backward(loss_node);

    auto out_dir = std::filesystem::path(cfg.dump_dir);
    write_meta(out_dir, cfg, scale);

    traced::save_tensor(q_host, (out_dir / "q.bin").string());
    traced::save_tensor(k_host, (out_dir / "k.bin").string());
    traced::save_tensor(v_host, (out_dir / "v.bin").string());
    traced::save_tensor(target_host, (out_dir / "target.bin").string());
    traced::save_tensor(attn_scores, (out_dir / "scores.bin").string(), &device);
    traced::save_tensor(attn_weights, (out_dir / "weights.bin").string(), &device);
    traced::save_tensor(attn_out, (out_dir / "out.bin").string(), &device);
    traced::save_tensor(d_q, (out_dir / "d_q.bin").string(), &device);
    traced::save_tensor(d_k, (out_dir / "d_k.bin").string(), &device);
    traced::save_tensor(d_v, (out_dir / "d_v.bin").string(), &device);

    auto loss_1d = ttnn::reshape(mse.loss, ttnn::Shape({1}));
    traced::save_tensor(loss_1d, (out_dir / "loss.bin").string(), &device);

    fmt::print("Saved outputs to {}\n", cfg.dump_dir);
    return 0;
}
