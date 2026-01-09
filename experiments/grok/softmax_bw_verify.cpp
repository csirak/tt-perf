// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Verify softmax backward math against PyTorch using BF16 tensors.

#include "autograd/nn.hpp"
#include "autograd/utils/config.hpp"
#include "autograd/utils/tensor_io.hpp"

#include <ttnn/device.hpp>
#include <ttnn/operations/data_movement/tilize/tilize.hpp>
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
    uint32_t batch = 256;
    uint32_t seq = 32;
    uint32_t seed = 1;
    uint32_t dump_outputs = 1;
    std::string dump_dir = "/home/howard/ttnn-perf/experiments/grok/outputs/softmax_bw";
};

Config load_config(const std::string& path) {
    Config cfg;
    auto kv = load_kv_file(path);
    get_u32(kv, "batch", cfg.batch);
    get_u32(kv, "seq", cfg.seq);
    get_u32(kv, "seed", cfg.seed);
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
    if (!file) return;
    file << "{\n";
    file << "  \"batch\": " << cfg.batch << ",\n";
    file << "  \"seq\": " << cfg.seq << ",\n";
    file << "  \"seed\": " << cfg.seed << "\n";
    file << "}\n";
}

}  // namespace

int main() {
    const char* env_path = std::getenv("SOFTMAX_BW_CONFIG");
    std::string config_path = env_path ? env_path
        : "/home/howard/ttnn-perf/experiments/grok/softmax_bw_verify.yaml";
    Config cfg = load_config(config_path);

    fmt::print("# Softmax BW verify config: {}\n", config_path);
    if (cfg.batch % 32 != 0 || cfg.seq % 32 != 0) {
        fmt::print("ERROR: batch and seq must be multiples of 32\n");
        return 1;
    }

    DeviceGuard guard;
    MeshDevice& device = guard.get();

    auto scores_host = make_host_bf16(
        make_random_vec(static_cast<size_t>(cfg.batch) * cfg.seq * cfg.seq, cfg.seed),
        ttnn::Shape({cfg.batch, cfg.seq, cfg.seq})
    );
    auto d_attn_host = make_host_bf16(
        make_random_vec(static_cast<size_t>(cfg.batch) * cfg.seq * cfg.seq, cfg.seed + 1),
        ttnn::Shape({cfg.batch, cfg.seq, cfg.seq})
    );

    Tensor scores = ttnn::tilize(scores_host.to_device(&device));
    Tensor d_attn = ttnn::tilize(d_attn_host.to_device(&device));

    Tensor weights = ttnn::softmax(scores, -1, std::nullopt, get_softmax_compute_config(), true);

    // Backward using dim = -1 (current implementation)
    auto dy_y_neg1 = ttnn::multiply(d_attn, weights);
    auto sum_dy_y_neg1 = ttnn::sum(dy_y_neg1, -1, true, std::nullopt, get_fp32_acc_compute_config());
    auto d_scores_neg1 = ttnn::multiply(weights, ttnn::subtract(d_attn, sum_dy_y_neg1));

    // Backward using dim = 2 (explicit last dim)
    auto dy_y_dim2 = ttnn::multiply(d_attn, weights);
    auto sum_dy_y_dim2 = ttnn::sum(dy_y_dim2, 2, true, std::nullopt, get_fp32_acc_compute_config());
    auto d_scores_dim2 = ttnn::multiply(weights, ttnn::subtract(d_attn, sum_dy_y_dim2));

    if (cfg.dump_outputs != 0) {
        write_meta(cfg.dump_dir, cfg);
        traced::save_tensor(scores_host, (std::filesystem::path(cfg.dump_dir) / "scores.bin").string());
        traced::save_tensor(d_attn_host, (std::filesystem::path(cfg.dump_dir) / "d_attn.bin").string());
        traced::save_tensor(weights, (std::filesystem::path(cfg.dump_dir) / "weights.bin").string(), &device);
        traced::save_tensor(d_scores_neg1, (std::filesystem::path(cfg.dump_dir) / "d_scores_neg1.bin").string(), &device);
        traced::save_tensor(d_scores_dim2, (std::filesystem::path(cfg.dump_dir) / "d_scores_dim2.bin").string(), &device);
    }

    fmt::print("# Done\n");
    return 0;
}
