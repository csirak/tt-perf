// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GELU backward verification (TTNN C++ vs PyTorch).

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
    uint32_t seed = 1;
    std::string dump_dir = "/home/howard/ttnn-perf/experiments/grok/outputs/gelu_bw";
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
    get_u32(kv, "seed", cfg.seed);
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

void write_meta(const std::filesystem::path& dir, const Config& cfg, const std::string& approx) {
    std::filesystem::create_directories(dir);
    std::ofstream file(dir / "meta.json");
    if (!file) {
        return;
    }
    file << "{\n";
    file << "  \"batch\": " << cfg.batch << ",\n";
    file << "  \"seq\": " << cfg.seq << ",\n";
    file << "  \"dim\": " << cfg.dim << ",\n";
    file << "  \"seed\": " << cfg.seed << ",\n";
    file << "  \"gelu_approx\": \"" << approx << "\"\n";
    file << "}\n";
}

}  // namespace

int main() {
    const char* env_path = std::getenv("GELU_BW_CONFIG");
    std::string config_path = env_path
        ? env_path
        : "/home/howard/ttnn-perf/experiments/grok/gelu_bw_verify.yaml";

    Config cfg = load_config(config_path);
    fmt::print("# GELU BW config: {}\n", config_path);
    fmt::print("# batch={}, seq={}, dim={}\n", cfg.batch, cfg.seq, cfg.dim);

    if (cfg.batch % 32 != 0 || cfg.seq % 32 != 0 || cfg.dim % 32 != 0) {
        fmt::print("ERROR: batch, seq, dim must be multiples of 32\n");
        return 1;
    }

    DeviceGuard guard;
    MeshDevice& device = guard.get();

    g_seed_counter = std::max<uint32_t>(1, cfg.seed);

    const size_t numel = static_cast<size_t>(cfg.batch) * cfg.seq * cfg.dim;
    auto x_data = make_random_vec(numel, cfg.seed);
    auto target_data = make_random_vec(numel, cfg.seed + 1);

    auto x_host = make_host_bf16(x_data, ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}));
    auto target_host = make_host_bf16(target_data, ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}));

    Tensor x = ttnn::tilize(x_host.to_device(&device));
    Tensor target = ttnn::tilize(target_host.to_device(&device));
    Tensor d_x = make_zeros(ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}), device);
    Tensor out = make_zeros(ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}), device);
    Tensor d_out = make_zeros(ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}), device);

    Graph graph;
    auto* x_node = graph.leaf(&x, &d_x, true);
    auto* out_node = gelu(graph, x_node, &out, &d_out);

    MeanSquaredError mse(cfg.batch, cfg.seq, cfg.dim, device);
    auto* target_node = graph.leaf(&target, nullptr, false);
    auto* loss_node = mse.build(graph, out_node, target_node);
    graph.build_topo(loss_node);

    mse.execute_forward(out, target);
    graph.zero_grad();
    graph.backward(loss_node);

    const char* approx_env = std::getenv("GELU_APPROX");
    std::string approx = (approx_env && *approx_env) ? approx_env : "none";

    auto out_dir = std::filesystem::path(cfg.dump_dir);
    write_meta(out_dir, cfg, approx);

    traced::save_tensor(x_host, (out_dir / "x.bin").string());
    traced::save_tensor(target_host, (out_dir / "target.bin").string());
    traced::save_tensor(out, (out_dir / "out.bin").string(), &device);
    traced::save_tensor(d_x, (out_dir / "d_x.bin").string(), &device);

    auto loss_1d = ttnn::reshape(mse.loss, ttnn::Shape({1}));
    traced::save_tensor(loss_1d, (out_dir / "loss.bin").string(), &device);

    fmt::print("Saved outputs to {}\n", cfg.dump_dir);
    return 0;
}
