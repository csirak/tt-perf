// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Grokking experiment (modular division) using static autograd + GPT-2.

#include "autograd/nn.hpp"
#include "autograd/utils/config.hpp"
#include "data/grok.hpp"
// example_autograd builds a single translation unit; include implementation.
#include "data/grok.cpp"

#include <ttnn/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>

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
    uint32_t p = 127;
    float train_frac = 0.5f;
    uint32_t pad_to = 32;
    uint32_t batch_size = 256;
    uint32_t steps = 200;
    uint32_t log_every = 10;
    uint32_t eval_every = 50;
    uint32_t seed = 0;
    float lr = 1e-3f;
    uint32_t dim = 128;
    uint32_t heads = 4;
    uint32_t ffn_mult = 4;
    uint32_t layers = 2;
    std::string csv_path = "";
};

Config load_config(const std::string& path) {
    Config cfg;
    auto kv = load_kv_file(path);
    if (kv.empty()) {
        fmt::print("# Config not found: {} (using defaults)\n", path);
        return cfg;
    }

    get_u32(kv, "p", cfg.p);
    get_f32(kv, "train_frac", cfg.train_frac);
    get_u32(kv, "pad_to", cfg.pad_to);
    get_u32(kv, "batch_size", cfg.batch_size);
    get_u32(kv, "steps", cfg.steps);
    get_u32(kv, "log_every", cfg.log_every);
    get_u32(kv, "eval_every", cfg.eval_every);
    get_u32(kv, "seed", cfg.seed);
    get_f32(kv, "lr", cfg.lr);
    get_u32(kv, "dim", cfg.dim);
    get_u32(kv, "heads", cfg.heads);
    get_u32(kv, "ffn_mult", cfg.ffn_mult);
    get_u32(kv, "layers", cfg.layers);
    get_str(kv, "csv", cfg.csv_path);

    return cfg;
}

void append_csv(const std::string& path, uint32_t step, float train_loss, float val_loss,
                double interval_s, double total_s) {
    if (path.empty()) {
        return;
    }
    std::ifstream in(path);
    bool write_header = !in.good();
    in.close();
    std::ofstream file(path, std::ios::app);
    if (!file) {
        return;
    }
    if (write_header) {
        file << "step,train_loss,val_loss,interval_s,total_s\n";
    }
    file << step << "," << train_loss << "," << val_loss << "," << interval_s << "," << total_s << "\n";
}

template <size_t N>
int run_train(const Config& cfg, MeshDevice& device) {
    grokking::ModularDivisionDataset dataset(cfg.p, cfg.train_frac, cfg.pad_to, cfg.seed);
    if (cfg.batch_size % 32 != 0 || cfg.pad_to % 32 != 0 || cfg.dim % 32 != 0 ||
        dataset.vocab_size % 32 != 0) {
        fmt::print("ERROR: batch, pad_to, dim, vocab must be multiples of 32\n");
        return 1;
    }
    if (cfg.dim % cfg.heads != 0) {
        fmt::print("ERROR: dim ({}) must be divisible by heads ({})\n", cfg.dim, cfg.heads);
        return 1;
    }

    PersistentGrokGPT2<N> model(cfg.batch_size, cfg.pad_to, cfg.dim, cfg.heads,
                                cfg.ffn_mult, dataset.vocab_size, cfg.lr, device);
    LastTokenCrossEntropy loss(cfg.batch_size, cfg.pad_to, dataset.vocab_size,
                               dataset.answer_pos, device);

    model.build_graph();
    auto* loss_node = loss.build(model.graph, model.logits_node);
    model.graph.build_topo(loss_node);

    fmt::print("# Grokking config: p={}, train_frac={}, pad_to={}, vocab={}, batch={}\n",
               cfg.p, cfg.train_frac, cfg.pad_to, dataset.vocab_size, cfg.batch_size);
    fmt::print("# Model: dim={}, heads={}, ffn_mult={}, layers={}, lr={}\n",
               cfg.dim, cfg.heads, cfg.ffn_mult, cfg.layers, cfg.lr);

    auto start = std::chrono::high_resolution_clock::now();
    auto last = start;

    for (uint32_t step = 1; step <= cfg.steps; ++step) {
        auto batch = dataset.sample_batch(cfg.batch_size, true);
        model.set_tokens(batch.tokens);

        {
#ifdef TRACY_ENABLE
            ZoneScopedN("forward");
#endif
            model.execute_forward();
            loss.execute_forward(model.output_proj.out, batch.targets);
            tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        }

        {
#ifdef TRACY_ENABLE
            ZoneScopedN("backward");
#endif
            model.graph.zero_grad();
            model.graph.backward(loss_node);
            tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        }

        {
#ifdef TRACY_ENABLE
            ZoneScopedN("sgd_step");
#endif
            model.sgd_step(cfg.lr);
            tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        }

        if (cfg.log_every > 0 && (step % cfg.log_every) == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double interval_s = std::chrono::duration<double>(now - last).count();
            double total_s = std::chrono::duration<double>(now - start).count();

            float train_loss = loss.get_loss();
            float val_loss = train_loss;
            if (cfg.eval_every > 0 && (step % cfg.eval_every) == 0) {
                auto val_batch = dataset.sample_batch(cfg.batch_size, false);
                model.set_tokens(val_batch.tokens);
                model.execute_forward();
                loss.execute_forward(model.output_proj.out, val_batch.targets);
                tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
                val_loss = loss.get_loss();
            }

            fmt::print("step {} train_loss {:.6f} val_loss {:.6f} interval_s {:.2f} total_s {:.2f}\n",
                       step, train_loss, val_loss, interval_s, total_s);
            append_csv(cfg.csv_path, step, train_loss, val_loss, interval_s, total_s);
            last = now;
        }
    }

    return 0;
}

}  // namespace

int main() {
    const char* env_path = std::getenv("GROK_CONFIG");
    std::string config_path = env_path ? env_path : "/home/howard/ttnn-perf/experiments/grok/default.yaml";
    Config cfg = load_config(config_path);

    fmt::print("# Grok config: {}\n", config_path);
    DeviceGuard guard;
    MeshDevice& device = guard.get();

    switch (cfg.layers) {
        case 1: return run_train<1>(cfg, device);
        case 2: return run_train<2>(cfg, device);
        case 3: return run_train<3>(cfg, device);
        case 6: return run_train<6>(cfg, device);
        default:
            fmt::print("ERROR: supported layers: 1, 2, 3, 6\n");
            return 1;
    }
}
