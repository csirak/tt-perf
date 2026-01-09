// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Embedding backward verification (TTNN C++ vs PyTorch).

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
    uint32_t vocab = 128;
    uint32_t dim = 128;
    uint32_t seed = 1;
    float init_std = 0.02f;
    std::string dump_dir = "/home/howard/ttnn-perf/experiments/grok/outputs/embedding_bw";
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
    get_u32(kv, "vocab", cfg.vocab);
    get_u32(kv, "dim", cfg.dim);
    get_u32(kv, "seed", cfg.seed);
    get_f32(kv, "init_std", cfg.init_std);
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

std::vector<uint32_t> make_random_indices(size_t n, uint32_t vocab, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dist(0, vocab - 1);
    std::vector<uint32_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = dist(gen);
    }
    return out;
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

void write_meta(const std::filesystem::path& dir, const Config& cfg) {
    std::filesystem::create_directories(dir);
    std::ofstream file(dir / "meta.json");
    if (!file) {
        return;
    }
    file << "{\n";
    file << "  \"batch\": " << cfg.batch << ",\n";
    file << "  \"seq\": " << cfg.seq << ",\n";
    file << "  \"vocab\": " << cfg.vocab << ",\n";
    file << "  \"dim\": " << cfg.dim << ",\n";
    file << "  \"seed\": " << cfg.seed << ",\n";
    file << "  \"init_std\": " << cfg.init_std << "\n";
    file << "}\n";
}

}  // namespace

int main() {
    const char* env_path = std::getenv("EMBEDDING_BW_CONFIG");
    std::string config_path = env_path
        ? env_path
        : "/home/howard/ttnn-perf/experiments/grok/embedding_bw_verify.yaml";

    Config cfg = load_config(config_path);
    fmt::print("# Embedding BW config: {}\n", config_path);
    fmt::print("# batch={}, seq={}, vocab={}, dim={}\n", cfg.batch, cfg.seq, cfg.vocab, cfg.dim);

    if (cfg.batch % 32 != 0 || cfg.seq % 32 != 0 || cfg.vocab % 32 != 0 || cfg.dim % 32 != 0) {
        fmt::print("ERROR: batch, seq, vocab, dim must be multiples of 32\n");
        return 1;
    }

    DeviceGuard guard;
    MeshDevice& device = guard.get();

    g_seed_counter = std::max<uint32_t>(1, cfg.seed);

    const size_t num_tokens = static_cast<size_t>(cfg.batch) * cfg.seq;
    auto indices = make_random_indices(num_tokens, cfg.vocab, cfg.seed);
    auto target_data = make_random_vec(num_tokens * cfg.dim, cfg.seed + 1);

    auto indices_tensor = make_indices(indices, ttnn::Shape({cfg.batch, cfg.seq}), device);
    auto target_host = make_host_bf16(target_data, ttnn::Shape({cfg.batch, cfg.seq, cfg.dim}));
    auto target = ttnn::tilize(target_host.to_device(&device));

    Embedding emb(cfg.vocab, cfg.dim, cfg.batch, cfg.seq, cfg.init_std, device);

    Graph graph;
    auto* idx_node = graph.leaf(&indices_tensor, nullptr, false);
    auto* out_node = emb.forward(graph, idx_node);

    auto* target_node = graph.leaf(&target, nullptr, false);
    MeanSquaredError mse(cfg.batch, cfg.seq, cfg.dim, device);
    auto* loss_node = mse.build(graph, out_node, target_node);
    graph.build_topo(loss_node);

    mse.execute_forward(*out_node->data, target);
    graph.zero_grad();
    graph.backward(loss_node);

    auto out_dir = std::filesystem::path(cfg.dump_dir);
    write_meta(out_dir, cfg);

    save_u32_tensor(indices, ttnn::Shape({cfg.batch, cfg.seq}), (out_dir / "indices_u32.bin").string());
    traced::save_tensor(target_host, (out_dir / "target.bin").string());
    traced::save_tensor(emb.weight, (out_dir / "weight.bin").string(), &device);
    traced::save_tensor(emb.out, (out_dir / "output.bin").string(), &device);
    traced::save_tensor(emb.d_weight, (out_dir / "weight_grad.bin").string(), &device);

    auto loss_1d = ttnn::reshape(mse.loss, ttnn::Shape({1}));
    traced::save_tensor(loss_1d, (out_dir / "loss.bin").string(), &device);

    fmt::print("Saved outputs to {}\n", cfg.dump_dir);
    return 0;
}
