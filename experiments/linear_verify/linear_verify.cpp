// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Incremental BF16 verification: Linear, Linear + LayerNorm, Linear + DyT.

#include "autograd/nn.hpp"
#include "autograd/utils/config.hpp"
#include "autograd/utils/tensor_io.hpp"

#include <ttnn/device.hpp>
#include <ttnn/operations/data_movement/tilize/tilize.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
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
    uint32_t batch_size = 256;
    uint32_t seq = 32;
    uint32_t in_dim = 128;
    uint32_t out_dim = 128;
    uint32_t stage = 0;
    uint32_t seed = 1;
    float eps = 1e-5f;
    float dyt_alpha = 1.0f;
    uint32_t dump_outputs = 1;
    std::string dump_dir = "/home/howard/ttnn-perf/experiments/linear_verify/outputs/cpp";
};

Config load_config(const std::string& path) {
    Config cfg;
    auto kv = load_kv_file(path);
    get_u32(kv, "batch_size", cfg.batch_size);
    get_u32(kv, "seq", cfg.seq);
    get_u32(kv, "in_dim", cfg.in_dim);
    get_u32(kv, "out_dim", cfg.out_dim);
    get_u32(kv, "stage", cfg.stage);
    get_u32(kv, "seed", cfg.seed);
    get_f32(kv, "eps", cfg.eps);
    get_f32(kv, "dyt_alpha", cfg.dyt_alpha);
    get_u32(kv, "dump_outputs", cfg.dump_outputs);
    get_str(kv, "dump_dir", cfg.dump_dir);
    return cfg;
}

std::vector<float> make_random_vec(size_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
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

void write_meta(const std::string& dir, const Config& cfg) {
    std::filesystem::create_directories(dir);
    std::ofstream file(std::filesystem::path(dir) / "meta.json");
    if (!file) {
        return;
    }
    file << "{\n";
    file << "  \"batch_size\": " << cfg.batch_size << ",\n";
    file << "  \"seq\": " << cfg.seq << ",\n";
    file << "  \"in_dim\": " << cfg.in_dim << ",\n";
    file << "  \"out_dim\": " << cfg.out_dim << ",\n";
    file << "  \"stage\": " << cfg.stage << ",\n";
    file << "  \"seed\": " << cfg.seed << ",\n";
    file << "  \"eps\": " << cfg.eps << ",\n";
    file << "  \"dyt_alpha\": " << cfg.dyt_alpha << "\n";
    file << "}\n";
}

}  // namespace

int main() {
    const char* env_path = std::getenv("LINEAR_VERIFY_CONFIG");
    std::string config_path = env_path ? env_path
        : "/home/howard/ttnn-perf/experiments/linear_verify/default.yaml";
    Config cfg = load_config(config_path);

    fmt::print("# Linear verify config: {}\n", config_path);
    if (cfg.batch_size % 32 != 0 || cfg.seq % 32 != 0 || cfg.in_dim % 32 != 0 || cfg.out_dim % 32 != 0) {
        fmt::print("ERROR: batch, seq, in_dim, out_dim must be multiples of 32\n");
        return 1;
    }
    if (cfg.stage > 2) {
        fmt::print("ERROR: stage must be 0 (Linear), 1 (Linear+LayerNorm), or 2 (Linear+DyT)\n");
        return 1;
    }

    DeviceGuard guard;
    MeshDevice& device = guard.get();

    g_seed_counter = std::max<uint32_t>(1, cfg.seed);

    auto input_data = make_random_vec(static_cast<size_t>(cfg.batch_size) * cfg.seq * cfg.in_dim, cfg.seed);
    auto target_data = make_random_vec(static_cast<size_t>(cfg.batch_size) * cfg.seq * cfg.out_dim, cfg.seed + 1);

    auto input_host = make_host_bf16(input_data, ttnn::Shape({cfg.batch_size, cfg.seq, cfg.in_dim}));
    auto target_host = make_host_bf16(target_data, ttnn::Shape({cfg.batch_size, cfg.seq, cfg.out_dim}));

    Tensor input = ttnn::tilize(input_host.to_device(&device));
    Tensor target = ttnn::tilize(target_host.to_device(&device));
    Tensor d_input = make_zeros(ttnn::Shape({cfg.batch_size, cfg.seq, cfg.in_dim}), device);

    Linear3D linear(cfg.batch_size, cfg.seq, cfg.in_dim, cfg.out_dim, 0.02f, device, InitKind::Randn);
    LayerNorm ln(cfg.batch_size, cfg.seq, cfg.out_dim, cfg.eps, device);
    std::unique_ptr<DyT> dyt;
    if (cfg.stage == 2) {
        dyt = std::make_unique<DyT>(cfg.batch_size, cfg.seq, cfg.out_dim, cfg.dyt_alpha, device);
    }
    Tensor ln_builtin;

    Graph graph;
    auto* x = graph.leaf(&input, &d_input, true);
    auto* t = graph.leaf(&target, nullptr, false);

    auto* h = linear.forward(graph, x);
    if (cfg.stage == 1) {
        h = ln.forward(graph, h);
        if (cfg.dump_outputs != 0) {
            auto compute_cfg = get_fp32_acc_compute_config();
            ln_builtin = ttnn::layer_norm(linear.out, cfg.eps, ln.gamma, ln.beta,
                                          std::nullopt, std::nullopt, std::nullopt,
                                          compute_cfg);
        }
    } else if (cfg.stage == 2) {
        h = dyt->forward(graph, h);
    }

    MeanSquaredError mse(cfg.batch_size, cfg.seq, cfg.out_dim, device);
    auto* loss_node = mse.build(graph, h, t);
    graph.build_topo(loss_node);

    mse.execute_forward(*h->data, target);

    if (cfg.dump_outputs != 0) {
        write_meta(cfg.dump_dir, cfg);
        traced::save_tensor(input_host, (std::filesystem::path(cfg.dump_dir) / "input.bin").string());
        traced::save_tensor(target_host, (std::filesystem::path(cfg.dump_dir) / "target.bin").string());
        traced::save_tensor(linear.weight, (std::filesystem::path(cfg.dump_dir) / "weight.bin").string(), &device);
        traced::save_tensor(linear.bias, (std::filesystem::path(cfg.dump_dir) / "bias.bin").string(), &device);
        traced::save_tensor(linear.out, (std::filesystem::path(cfg.dump_dir) / "linear_out.bin").string(), &device);
        if (cfg.stage == 1) {
            traced::save_tensor(ln.gamma, (std::filesystem::path(cfg.dump_dir) / "ln_gamma.bin").string(), &device);
            traced::save_tensor(ln.beta, (std::filesystem::path(cfg.dump_dir) / "ln_beta.bin").string(), &device);
            traced::save_tensor(ln.out, (std::filesystem::path(cfg.dump_dir) / "ln_out.bin").string(), &device);
            traced::save_tensor(ln_builtin, (std::filesystem::path(cfg.dump_dir) / "ln_out_builtin.bin").string(), &device);
        } else if (cfg.stage == 2) {
            traced::save_tensor(dyt->alpha, (std::filesystem::path(cfg.dump_dir) / "dyt_alpha.bin").string(), &device);
            traced::save_tensor(dyt->gamma, (std::filesystem::path(cfg.dump_dir) / "dyt_gamma.bin").string(), &device);
            traced::save_tensor(dyt->beta, (std::filesystem::path(cfg.dump_dir) / "dyt_beta.bin").string(), &device);
            traced::save_tensor(dyt->out, (std::filesystem::path(cfg.dump_dir) / "dyt_out.bin").string(), &device);
        }
        traced::save_tensor(mse.loss, (std::filesystem::path(cfg.dump_dir) / "loss.bin").string(), &device);
    }

    graph.zero_grad();
    graph.backward_scaled(loss_node, 1.0f);
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    if (cfg.dump_outputs != 0) {
        traced::save_tensor(d_input, (std::filesystem::path(cfg.dump_dir) / "input_grad.bin").string(), &device);
        traced::save_tensor(linear.d_weight, (std::filesystem::path(cfg.dump_dir) / "weight_grad.bin").string(), &device);
        traced::save_tensor(linear.d_bias, (std::filesystem::path(cfg.dump_dir) / "bias_grad.bin").string(), &device);
        traced::save_tensor(linear.d_out, (std::filesystem::path(cfg.dump_dir) / "linear_out_grad.bin").string(), &device);
        if (cfg.stage == 1) {
            traced::save_tensor(ln.d_gamma, (std::filesystem::path(cfg.dump_dir) / "ln_gamma_grad.bin").string(), &device);
            traced::save_tensor(ln.d_beta, (std::filesystem::path(cfg.dump_dir) / "ln_beta_grad.bin").string(), &device);
            traced::save_tensor(ln.d_out, (std::filesystem::path(cfg.dump_dir) / "ln_out_grad.bin").string(), &device);
        } else if (cfg.stage == 2) {
            traced::save_tensor(dyt->d_alpha, (std::filesystem::path(cfg.dump_dir) / "dyt_alpha_grad.bin").string(), &device);
            traced::save_tensor(dyt->d_gamma, (std::filesystem::path(cfg.dump_dir) / "dyt_gamma_grad.bin").string(), &device);
            traced::save_tensor(dyt->d_beta, (std::filesystem::path(cfg.dump_dir) / "dyt_beta_grad.bin").string(), &device);
            traced::save_tensor(dyt->d_out, (std::filesystem::path(cfg.dump_dir) / "dyt_out_grad.bin").string(), &device);
        }
    }

    fmt::print("# Loss: {:.6f}\n", mse.get_loss());
    return 0;
}
