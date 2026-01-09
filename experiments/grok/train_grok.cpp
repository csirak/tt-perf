// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Grokking experiment (modular division) using static autograd + GPT-2.

#include "autograd/nn.hpp"
#include "autograd/utils/config.hpp"
#include "autograd/utils/tensor_io.hpp"
#include "data/grok.hpp"
// example_autograd builds a single translation unit; include implementation.
#include "data/grok.cpp"

#include <ttnn/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <system_error>
#include <string>
#include <type_traits>
#include <utility>

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

template <typename T, typename = void>
struct has_ln_final : std::false_type {};

template <typename T>
struct has_ln_final<T, std::void_t<decltype(std::declval<T>().ln_final)>> : std::true_type {};

struct Config {
    uint32_t p = 127;
    float train_frac = 0.5f;
    uint32_t pad_to = 32;
    uint32_t batch_size = 256;
    uint32_t steps = 200;
    uint32_t log_every = 10;
    uint32_t eval_every = 50;
    uint32_t seed = 0;
    // Optimizer config
    std::string optimizer = "sgd";  // "sgd" or "adam"
    float lr = 1e-3f;
    float momentum = 0.0f;
    float weight_decay = 0.0f;
    // Adam-specific config
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    // Model config
    uint32_t dim = 128;
    uint32_t heads = 4;
    uint32_t ffn_mult = 4;
    uint32_t layers = 2;
    float loss_scale = 1.0f;
    uint32_t use_layer_norm = 0;
    float ln_eps = 1e-5f;
    uint32_t dump_outputs = 0;
    uint32_t dump_steps = 0;
    std::string dump_dir = "/home/howard/ttnn-perf/experiments/grok/outputs/cpp";
    std::string dump_steps_dir = "/home/howard/ttnn-perf/experiments/grok/outputs/steps";
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
    get_str(kv, "optimizer", cfg.optimizer);
    get_f32(kv, "lr", cfg.lr);
    get_f32(kv, "momentum", cfg.momentum);
    get_f32(kv, "weight_decay", cfg.weight_decay);
    get_f32(kv, "beta1", cfg.beta1);
    get_f32(kv, "beta2", cfg.beta2);
    get_f32(kv, "eps", cfg.eps);
    get_u32(kv, "dim", cfg.dim);
    get_u32(kv, "heads", cfg.heads);
    get_u32(kv, "ffn_mult", cfg.ffn_mult);
    get_u32(kv, "layers", cfg.layers);
    get_f32(kv, "loss_scale", cfg.loss_scale);
    get_u32(kv, "use_layer_norm", cfg.use_layer_norm);
    get_f32(kv, "ln_eps", cfg.ln_eps);
    get_u32(kv, "dump_outputs", cfg.dump_outputs);
    get_u32(kv, "dump_steps", cfg.dump_steps);
    get_str(kv, "dump_dir", cfg.dump_dir);
    get_str(kv, "dump_steps_dir", cfg.dump_steps_dir);
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
    const bool is_tsv = path.size() >= 4 && path.rfind(".tsv") == (path.size() - 4);
    const char delim = is_tsv ? '\t' : ',';
    const double avg_step_s = (step > 0) ? (total_s / static_cast<double>(step)) : 0.0;

    if (write_header) {
        file << "step" << delim
             << "train_loss" << delim
             << "val_loss" << delim
             << "interval_s" << delim
             << "total_s" << delim
             << "avg_step_s\n";
    }
    file << step << delim
         << train_loss << delim
         << val_loss << delim
         << interval_s << delim
         << total_s << delim
         << avg_step_s << "\n";
}

void init_csv(const std::string& path) {
    if (path.empty()) {
        return;
    }
    const bool is_tsv = path.size() >= 4 && path.rfind(".tsv") == (path.size() - 4);
    const char delim = is_tsv ? '\t' : ',';
    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        return;
    }
    file << "step" << delim
         << "train_loss" << delim
         << "val_loss" << delim
         << "interval_s" << delim
         << "total_s" << delim
         << "avg_step_s\n";
}

void clear_step_files(const std::string& dir) {
    if (dir.empty()) {
        return;
    }
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) {
        return;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const auto name = entry.path().filename().string();
        if (name.rfind("step_", 0) == 0) {
            std::filesystem::remove(entry.path(), ec);
        }
    }
}

Tensor make_host_bf16(const std::vector<uint32_t>& data, const ttnn::Shape& shape) {
    std::vector<bfloat16> bf(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        bf[i] = bfloat16(static_cast<float>(data[i]));
    }
    auto layout = ttnn::TensorLayout(
        ttnn::DataType::BFLOAT16,
        ttnn::PageConfig(ttnn::ROW_MAJOR_LAYOUT),
        tt::tt_metal::MemoryConfig{}
    );
    return Tensor::from_vector(bf, ttnn::TensorSpec(shape, layout));
}

void write_meta(const std::string& dir, const Config& cfg, const grokking::ModularDivisionDataset& dataset) {
    std::filesystem::create_directories(dir);
    std::ofstream file(std::filesystem::path(dir) / "meta.json");
    if (!file) {
        return;
    }
    file << "{\n";
    file << "  \"p\": " << cfg.p << ",\n";
    file << "  \"train_frac\": " << cfg.train_frac << ",\n";
    file << "  \"pad_to\": " << cfg.pad_to << ",\n";
    file << "  \"batch_size\": " << cfg.batch_size << ",\n";
    file << "  \"dim\": " << cfg.dim << ",\n";
    file << "  \"heads\": " << cfg.heads << ",\n";
    file << "  \"ffn_mult\": " << cfg.ffn_mult << ",\n";
    file << "  \"layers\": " << cfg.layers << ",\n";
    file << "  \"loss_scale\": " << cfg.loss_scale << ",\n";
    file << "  \"use_layer_norm\": " << cfg.use_layer_norm << ",\n";
    file << "  \"ln_eps\": " << cfg.ln_eps << ",\n";
    file << "  \"lr\": " << cfg.lr << ",\n";
    file << "  \"vocab\": " << dataset.vocab_size << ",\n";
    file << "  \"answer_pos\": " << dataset.answer_pos << "\n";
    file << "}\n";
}

void save_norm_params(const std::filesystem::path& dir,
                      const std::string& prefix,
                      DyT& ln,
                      MeshDevice& device);

void save_norm_params(const std::filesystem::path& dir,
                      const std::string& prefix,
                      LayerNorm& ln,
                      MeshDevice& device);

void save_norm_grads(const std::filesystem::path& dir,
                     const std::string& prefix,
                     DyT& ln,
                     MeshDevice& device);

void save_norm_grads(const std::filesystem::path& dir,
                     const std::string& prefix,
                     LayerNorm& ln,
                     MeshDevice& device);

void accum_norm_sumsq(DyT& ln, double& sum);
void accum_norm_sumsq(LayerNorm& ln, double& sum);

template <size_t N, typename Model>
void dump_grok_outputs(const std::string& dir,
                       const Config& cfg,
                       const grokking::ModularDivisionDataset& dataset,
                       Model& model,
                       LastTokenCrossEntropy& loss,
                       const grokking::Batch& batch,
                       MeshDevice& device) {
    write_meta(dir, cfg, dataset);
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    auto tokens_tensor = make_host_bf16(batch.tokens, ttnn::Shape({cfg.batch_size, cfg.pad_to}));
    auto targets_tensor = make_host_bf16(batch.targets, ttnn::Shape({cfg.batch_size}));
    traced::save_tensor(tokens_tensor, (std::filesystem::path(dir) / "tokens.bin").string());
    traced::save_tensor(targets_tensor, (std::filesystem::path(dir) / "targets.bin").string());

    traced::save_tensor(model.tok.weight, (std::filesystem::path(dir) / "tok_weight.bin").string(), &device);
    traced::save_tensor(model.pos.weight, (std::filesystem::path(dir) / "pos_weight.bin").string(), &device);
    traced::save_tensor(model.output_proj.weight, (std::filesystem::path(dir) / "output_weight.bin").string(), &device);
    traced::save_tensor(model.output_proj.bias, (std::filesystem::path(dir) / "output_bias.bin").string(), &device);

    traced::save_tensor(model.tok_plus_pos, (std::filesystem::path(dir) / "tok_plus_pos.bin").string(), &device);

    for (size_t i = 0; i < N; ++i) {
        auto& layer = model.layers[i];
        auto prefix = fmt::format("layer{}", i);
        auto base = std::filesystem::path(dir);
        save_norm_params(base, prefix + "_ln1", layer.ln1, device);
        save_norm_params(base, prefix + "_ln2", layer.ln2, device);

        traced::save_tensor(layer.wq.weight, (std::filesystem::path(dir) / (prefix + "_wq_weight.bin")).string(), &device);
        traced::save_tensor(layer.wq.bias, (std::filesystem::path(dir) / (prefix + "_wq_bias.bin")).string(), &device);
        traced::save_tensor(layer.wk.weight, (std::filesystem::path(dir) / (prefix + "_wk_weight.bin")).string(), &device);
        traced::save_tensor(layer.wk.bias, (std::filesystem::path(dir) / (prefix + "_wk_bias.bin")).string(), &device);
        traced::save_tensor(layer.wv.weight, (std::filesystem::path(dir) / (prefix + "_wv_weight.bin")).string(), &device);
        traced::save_tensor(layer.wv.bias, (std::filesystem::path(dir) / (prefix + "_wv_bias.bin")).string(), &device);
        traced::save_tensor(layer.wo.weight, (std::filesystem::path(dir) / (prefix + "_wo_weight.bin")).string(), &device);
        traced::save_tensor(layer.wo.bias, (std::filesystem::path(dir) / (prefix + "_wo_bias.bin")).string(), &device);

        traced::save_tensor(layer.ffn.w1.weight, (std::filesystem::path(dir) / (prefix + "_ffn_w1_weight.bin")).string(), &device);
        traced::save_tensor(layer.ffn.w1.bias, (std::filesystem::path(dir) / (prefix + "_ffn_w1_bias.bin")).string(), &device);
        traced::save_tensor(layer.ffn.w2.weight, (std::filesystem::path(dir) / (prefix + "_ffn_w2_weight.bin")).string(), &device);
        traced::save_tensor(layer.ffn.w2.bias, (std::filesystem::path(dir) / (prefix + "_ffn_w2_bias.bin")).string(), &device);

        Tensor* layer_input = (i == 0) ? &model.tok_plus_pos : &model.layers[i - 1].output;
        traced::save_tensor(*layer_input, (std::filesystem::path(dir) / (prefix + "_input.bin")).string(), &device);
        traced::save_tensor(layer.ln1.out, (std::filesystem::path(dir) / (prefix + "_ln1.bin")).string(), &device);
        traced::save_tensor(layer.wq.out, (std::filesystem::path(dir) / (prefix + "_wq.bin")).string(), &device);
        traced::save_tensor(layer.wk.out, (std::filesystem::path(dir) / (prefix + "_wk.bin")).string(), &device);
        traced::save_tensor(layer.wv.out, (std::filesystem::path(dir) / (prefix + "_wv.bin")).string(), &device);
        traced::save_tensor(layer.attn_scores, (std::filesystem::path(dir) / (prefix + "_attn_scores.bin")).string(), &device);
        traced::save_tensor(layer.attn_weights, (std::filesystem::path(dir) / (prefix + "_attn_weights.bin")).string(), &device);
        traced::save_tensor(layer.attn_out, (std::filesystem::path(dir) / (prefix + "_attn_out.bin")).string(), &device);
        traced::save_tensor(layer.wo.out, (std::filesystem::path(dir) / (prefix + "_wo.bin")).string(), &device);
        traced::save_tensor(layer.residual1, (std::filesystem::path(dir) / (prefix + "_residual1.bin")).string(), &device);
        traced::save_tensor(layer.ln2.out, (std::filesystem::path(dir) / (prefix + "_ln2.bin")).string(), &device);
        traced::save_tensor(layer.ffn.w1.out, (std::filesystem::path(dir) / (prefix + "_ffn_w1.bin")).string(), &device);
        traced::save_tensor(layer.ffn.gelu_out, (std::filesystem::path(dir) / (prefix + "_ffn_gelu.bin")).string(), &device);
        traced::save_tensor(layer.ffn.w2.out, (std::filesystem::path(dir) / (prefix + "_ffn_w2.bin")).string(), &device);
        traced::save_tensor(layer.output, (std::filesystem::path(dir) / (prefix + "_output.bin")).string(), &device);
    }

    if constexpr (has_ln_final<Model>::value) {
        auto base = std::filesystem::path(dir);
        save_norm_params(base, "ln_final", model.ln_final, device);
        traced::save_tensor(model.ln_final.out, (base / "ln_final.bin").string(), &device);
    }

    traced::save_tensor(model.output_proj.out, (std::filesystem::path(dir) / "logits.bin").string(), &device);
    auto loss_1d = ttnn::reshape(loss.loss, ttnn::Shape({1}));
    traced::save_tensor(loss_1d, (std::filesystem::path(dir) / "loss.bin").string(), &device);
}

void ensure_dir(const std::string& dir) {
    std::filesystem::create_directories(dir);
}

double sumsq_bf16(const Tensor& t) {
    auto cpu_tensor = t.cpu();
    auto data = cpu_tensor.to_vector<bfloat16>();
    double sum = 0.0;
    for (auto v : data) {
        float f = static_cast<float>(v);
        sum += static_cast<double>(f) * static_cast<double>(f);
    }
    return sum;
}

void save_norm_params(const std::filesystem::path& dir,
                      const std::string& prefix,
                      DyT& ln,
                      MeshDevice& device) {
    traced::save_tensor(ln.alpha, (dir / (prefix + "_alpha.bin")).string(), &device);
    traced::save_tensor(ln.gamma, (dir / (prefix + "_gamma.bin")).string(), &device);
    traced::save_tensor(ln.beta, (dir / (prefix + "_beta.bin")).string(), &device);
}

void save_norm_params(const std::filesystem::path& dir,
                      const std::string& prefix,
                      LayerNorm& ln,
                      MeshDevice& device) {
    traced::save_tensor(ln.gamma, (dir / (prefix + "_gamma.bin")).string(), &device);
    traced::save_tensor(ln.beta, (dir / (prefix + "_beta.bin")).string(), &device);
}

void save_norm_grads(const std::filesystem::path& dir,
                     const std::string& prefix,
                     DyT& ln,
                     MeshDevice& device) {
    traced::save_tensor(ln.d_alpha, (dir / (prefix + "_alpha_grad.bin")).string(), &device);
    traced::save_tensor(ln.d_gamma, (dir / (prefix + "_gamma_grad.bin")).string(), &device);
    traced::save_tensor(ln.d_beta, (dir / (prefix + "_beta_grad.bin")).string(), &device);
}

void save_norm_grads(const std::filesystem::path& dir,
                     const std::string& prefix,
                     LayerNorm& ln,
                     MeshDevice& device) {
    traced::save_tensor(ln.d_gamma, (dir / (prefix + "_gamma_grad.bin")).string(), &device);
    traced::save_tensor(ln.d_beta, (dir / (prefix + "_beta_grad.bin")).string(), &device);
}

void accum_norm_sumsq(DyT& ln, double& sum) {
    sum += sumsq_bf16(ln.d_alpha);
    sum += sumsq_bf16(ln.d_gamma);
    sum += sumsq_bf16(ln.d_beta);
}

void accum_norm_sumsq(LayerNorm& ln, double& sum) {
    sum += sumsq_bf16(ln.d_gamma);
    sum += sumsq_bf16(ln.d_beta);
}

double sumabs_bf16(const Tensor& t) {
    auto cpu_tensor = t.cpu();
    auto data = cpu_tensor.to_vector<bfloat16>();
    double sum = 0.0;
    for (auto v : data) {
        sum += std::abs(static_cast<float>(v));
    }
    return sum;
}

double l2norm_bf16(const Tensor& t) {
    return std::sqrt(sumsq_bf16(t));
}

template <size_t N, typename Model>
double grad_sumsq(Model& model) {
    double sum = 0.0;
    sum += sumsq_bf16(model.tok.d_weight);
    sum += sumsq_bf16(model.pos.d_weight);
    sum += sumsq_bf16(model.output_proj.d_weight);
    sum += sumsq_bf16(model.output_proj.d_bias);
    for (size_t i = 0; i < N; ++i) {
        auto& layer = model.layers[i];
        accum_norm_sumsq(layer.ln1, sum);
        accum_norm_sumsq(layer.ln2, sum);
        sum += sumsq_bf16(layer.wq.d_weight);
        sum += sumsq_bf16(layer.wq.d_bias);
        sum += sumsq_bf16(layer.wk.d_weight);
        sum += sumsq_bf16(layer.wk.d_bias);
        sum += sumsq_bf16(layer.wv.d_weight);
        sum += sumsq_bf16(layer.wv.d_bias);
        sum += sumsq_bf16(layer.wo.d_weight);
        sum += sumsq_bf16(layer.wo.d_bias);
        sum += sumsq_bf16(layer.ffn.w1.d_weight);
        sum += sumsq_bf16(layer.ffn.w1.d_bias);
        sum += sumsq_bf16(layer.ffn.w2.d_weight);
        sum += sumsq_bf16(layer.ffn.w2.d_bias);
    }
    return sum;
}

void append_loss_tsv(const std::string& path, uint32_t step, float loss) {
    std::ifstream in(path);
    bool write_header = !in.good();
    in.close();
    std::ofstream file(path, std::ios::app);
    if (!file) {
        return;
    }
    if (write_header) {
        file << "step\tloss\n";
    }
    file << step << "\t" << loss << "\n";
}

void append_grad_tsv(const std::string& path, uint32_t step, float grad_norm) {
    std::ifstream in(path);
    bool write_header = !in.good();
    in.close();
    std::ofstream file(path, std::ios::app);
    if (!file) {
        return;
    }
    if (write_header) {
        file << "step\tgrad_norm\n";
    }
    file << step << "\t" << grad_norm << "\n";
}

template <typename Model>
void append_act_norms_tsv(const std::string& path, uint32_t step, Model& model) {
    std::ifstream in(path);
    bool write_header = !in.good();
    in.close();

    std::ofstream file(path, std::ios::app);
    if (!file) {
        return;
    }

    if (write_header) {
        file << "step";
        file << "\ttok_plus_pos_l2\ttok_plus_pos_l1";
        for (size_t i = 0; i < model.layers.size(); ++i) {
            file << "\tlayer" << i << "_ln1_l2\tlayer" << i << "_ln1_l1";
            file << "\tlayer" << i << "_attn_out_l2\tlayer" << i << "_attn_out_l1";
            file << "\tlayer" << i << "_ln2_l2\tlayer" << i << "_ln2_l1";
            file << "\tlayer" << i << "_ffn_w2_l2\tlayer" << i << "_ffn_w2_l1";
            file << "\tlayer" << i << "_output_l2\tlayer" << i << "_output_l1";
        }
        file << "\tlogits_l2\tlogits_l1\n";
    }

    auto write_norms = [&](const Tensor& t) {
        file << "\t" << l2norm_bf16(t) << "\t" << sumabs_bf16(t);
    };

    file << step;
    write_norms(model.tok_plus_pos);
    for (size_t i = 0; i < model.layers.size(); ++i) {
        auto& layer = model.layers[i];
        write_norms(layer.ln1.out);
        write_norms(layer.attn_out);
        write_norms(layer.ln2.out);
        write_norms(layer.ffn.w2.out);
        write_norms(layer.output);
    }
    write_norms(model.output_proj.out);
    file << "\n";
}

template <size_t N, typename Model>
void dump_grok_gradients(const std::string& dir,
                         Model& model,
                         MeshDevice& device) {
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    traced::save_tensor(model.tok.d_weight, (std::filesystem::path(dir) / "tok_weight_grad.bin").string(), &device);
    traced::save_tensor(model.pos.d_weight, (std::filesystem::path(dir) / "pos_weight_grad.bin").string(), &device);
    traced::save_tensor(model.d_tok_plus_pos,
                        (std::filesystem::path(dir) / "tok_plus_pos_grad.bin").string(), &device);

    for (size_t i = 0; i < N; ++i) {
        auto& layer = model.layers[i];
        auto prefix = fmt::format("layer{}", i);
        auto base = std::filesystem::path(dir);
        save_norm_grads(base, prefix + "_ln1", layer.ln1, device);
        save_norm_grads(base, prefix + "_ln2", layer.ln2, device);

        traced::save_tensor(layer.wq.d_weight,
                            (std::filesystem::path(dir) / (prefix + "_wq_weight_grad.bin")).string(), &device);
        traced::save_tensor(layer.wq.d_bias,
                            (std::filesystem::path(dir) / (prefix + "_wq_bias_grad.bin")).string(), &device);
        traced::save_tensor(layer.wk.d_weight,
                            (std::filesystem::path(dir) / (prefix + "_wk_weight_grad.bin")).string(), &device);
        traced::save_tensor(layer.wk.d_bias,
                            (std::filesystem::path(dir) / (prefix + "_wk_bias_grad.bin")).string(), &device);
        traced::save_tensor(layer.wv.d_weight,
                            (std::filesystem::path(dir) / (prefix + "_wv_weight_grad.bin")).string(), &device);
        traced::save_tensor(layer.wv.d_bias,
                            (std::filesystem::path(dir) / (prefix + "_wv_bias_grad.bin")).string(), &device);
        traced::save_tensor(layer.wo.d_weight,
                            (std::filesystem::path(dir) / (prefix + "_wo_weight_grad.bin")).string(), &device);
        traced::save_tensor(layer.wo.d_bias,
                            (std::filesystem::path(dir) / (prefix + "_wo_bias_grad.bin")).string(), &device);

        traced::save_tensor(layer.ffn.w1.d_weight,
                            (std::filesystem::path(dir) / (prefix + "_ffn_w1_weight_grad.bin")).string(), &device);
        traced::save_tensor(layer.ffn.w1.d_bias,
                            (std::filesystem::path(dir) / (prefix + "_ffn_w1_bias_grad.bin")).string(), &device);
        traced::save_tensor(layer.ffn.w2.d_weight,
                            (std::filesystem::path(dir) / (prefix + "_ffn_w2_weight_grad.bin")).string(), &device);
        traced::save_tensor(layer.ffn.w2.d_bias,
                            (std::filesystem::path(dir) / (prefix + "_ffn_w2_bias_grad.bin")).string(), &device);
    }

    if constexpr (has_ln_final<Model>::value) {
        save_norm_grads(std::filesystem::path(dir), "ln_final", model.ln_final, device);
    }

    traced::save_tensor(model.output_proj.d_weight,
                        (std::filesystem::path(dir) / "output_weight_grad.bin").string(), &device);
    traced::save_tensor(model.output_proj.d_bias,
                        (std::filesystem::path(dir) / "output_bias_grad.bin").string(), &device);
}

void dump_step_batch(const std::string& dir, uint32_t step,
                     const grokking::Batch& batch, const Config& cfg) {
    auto tokens_tensor = make_host_bf16(batch.tokens, ttnn::Shape({cfg.batch_size, cfg.pad_to}));
    auto targets_tensor = make_host_bf16(batch.targets, ttnn::Shape({cfg.batch_size}));
    auto step_prefix = fmt::format("step_{:04d}", step);
    traced::save_tensor(tokens_tensor, (std::filesystem::path(dir) / (step_prefix + "_tokens.bin")).string());
    traced::save_tensor(targets_tensor, (std::filesystem::path(dir) / (step_prefix + "_targets.bin")).string());
}

template <size_t N, typename ModelFactory>
int run_train(const Config& cfg, MeshDevice& device, ModelFactory make_model) {
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

    auto model = make_model(dataset);
    LastTokenCrossEntropy loss(cfg.batch_size, cfg.pad_to, dataset.vocab_size,
                               dataset.answer_pos, device);

    model.build_graph();
    auto* loss_node = loss.build(model.graph, model.logits_node);
    model.graph.build_topo(loss_node);

    fmt::print("# Grokking config: p={}, train_frac={}, pad_to={}, vocab={}, batch={}\n",
               cfg.p, cfg.train_frac, cfg.pad_to, dataset.vocab_size, cfg.batch_size);
    fmt::print("# Model: dim={}, heads={}, ffn_mult={}, layers={}, norm={}\n",
               cfg.dim, cfg.heads, cfg.ffn_mult, cfg.layers,
               cfg.use_layer_norm ? "layer_norm" : "dyt");

    // Setup optimizer
    bool use_adam = (cfg.optimizer == "adam" || cfg.optimizer == "adamw");
    std::unique_ptr<Adam> adam_opt;
    if (use_adam) {
        adam_opt = std::make_unique<Adam>(cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);
        model.register_adam(*adam_opt);
        fmt::print("# Optimizer: Adam (lr={}, beta1={}, beta2={}, eps={}, wd={})\n",
                   cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);
    } else {
        fmt::print("# Optimizer: SGD (lr={}, momentum={}, wd={})\n",
                   cfg.lr, cfg.momentum, cfg.weight_decay);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto last = start;
    bool dumped = false;
    bool dumped_grads = false;
    const auto loss_steps_path = (std::filesystem::path(cfg.dump_steps_dir) / "loss_cpp.tsv").string();
    const auto grad_steps_path = (std::filesystem::path(cfg.dump_steps_dir) / "grad_norm_cpp.tsv").string();
    const auto act_steps_path = (std::filesystem::path(cfg.dump_steps_dir) / "act_norms_cpp.tsv").string();
    if (cfg.dump_steps > 0) {
        ensure_dir(cfg.dump_steps_dir);
        clear_step_files(cfg.dump_steps_dir);
        std::filesystem::remove(loss_steps_path);
        std::filesystem::remove(grad_steps_path);
        std::filesystem::remove(act_steps_path);
    }
    init_csv(cfg.csv_path);

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

        if (!dumped && cfg.dump_outputs != 0) {
            dump_grok_outputs<N>(cfg.dump_dir, cfg, dataset, model, loss, batch, device);
            dumped = true;
        }

        if (cfg.dump_steps > 0 && step <= cfg.dump_steps) {
            dump_step_batch(cfg.dump_steps_dir, step, batch, cfg);
            append_loss_tsv(loss_steps_path, step, loss.get_loss());
            append_act_norms_tsv(act_steps_path, step, model);
        }

        {
#ifdef TRACY_ENABLE
            ZoneScopedN("backward");
#endif
            model.graph.zero_grad();
            model.graph.backward_scaled(loss_node, cfg.loss_scale);
            tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        }

        if (cfg.dump_steps > 0 && step <= cfg.dump_steps) {
            double sum = grad_sumsq<N>(model);
            append_grad_tsv(grad_steps_path, step, static_cast<float>(std::sqrt(sum)));
        }

        if (!dumped_grads && cfg.dump_outputs != 0) {
            dump_grok_gradients<N>(cfg.dump_dir, model, device);
            dumped_grads = true;
        }

        // Debug: track weight changes
        double tok_before = 0, out_before = 0;
        std::vector<bfloat16> tok_w_before, out_w_before;
        if (step <= 5) {
            tok_w_before = model.tok.weight.cpu().template to_vector<bfloat16>();
            out_w_before = model.output_proj.weight.cpu().template to_vector<bfloat16>();
            for (auto v : tok_w_before) tok_before += static_cast<float>(v) * static_cast<float>(v);
            for (auto v : out_w_before) out_before += static_cast<float>(v) * static_cast<float>(v);

            double tok_grad = sumsq_bf16(model.tok.d_weight);
            double out_grad = sumsq_bf16(model.output_proj.d_weight);
            fmt::print("Step {}: loss={:.6f}\n", step, loss.get_loss());
            fmt::print("  tok_grad_norm={:.6f}\n", std::sqrt(tok_grad));
            fmt::print("  out_grad_norm={:.6f}\n", std::sqrt(out_grad));
            if (step == 1) {
                fmt::print("  model.tok.weight addr={}\n", (void*)&model.tok.weight);
            }
        }

        {
#ifdef TRACY_ENABLE
            ZoneScopedN("optimizer_step");
#endif
            if (use_adam) {
                adam_opt->step();
            } else {
                model.sgd_step(cfg.lr, cfg.momentum, cfg.weight_decay);
            }
            tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        }

        // Debug: compute weight change after optimizer step
        if (step <= 5) {
            auto tok_w_after = model.tok.weight.cpu().template to_vector<bfloat16>();
            auto out_w_after = model.output_proj.weight.cpu().template to_vector<bfloat16>();

            double tok_diff = 0, out_diff = 0;
            for (size_t i = 0; i < tok_w_before.size(); ++i) {
                float d = static_cast<float>(tok_w_after[i]) - static_cast<float>(tok_w_before[i]);
                tok_diff += d * d;
            }
            for (size_t i = 0; i < out_w_before.size(); ++i) {
                float d = static_cast<float>(out_w_after[i]) - static_cast<float>(out_w_before[i]);
                out_diff += d * d;
            }

            double tok_after = 0, out_after = 0;
            for (auto v : tok_w_after) tok_after += static_cast<float>(v) * static_cast<float>(v);
            for (auto v : out_w_after) out_after += static_cast<float>(v) * static_cast<float>(v);

            fmt::print("  tok_weight_change={:.6f}\n", std::sqrt(tok_diff));
            fmt::print("  out_weight_change={:.6f}\n", std::sqrt(out_diff));
            fmt::print("  tok_norm: {:.2f} -> {:.2f}\n", std::sqrt(tok_before), std::sqrt(tok_after));
            fmt::print("  out_norm: {:.2f} -> {:.2f}\n", std::sqrt(out_before), std::sqrt(out_after));
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

template <size_t N>
int run_train_select(const Config& cfg, MeshDevice& device) {
    if (cfg.use_layer_norm != 0) {
        return run_train<N>(cfg, device, [&](const grokking::ModularDivisionDataset& dataset) {
            return PersistentGrokGPT2LN<N>(cfg.batch_size, cfg.pad_to, cfg.dim, cfg.heads,
                                           cfg.ffn_mult, dataset.vocab_size, cfg.lr, cfg.ln_eps, device);
        });
    }

    return run_train<N>(cfg, device, [&](const grokking::ModularDivisionDataset& dataset) {
        return PersistentGrokGPT2<N>(cfg.batch_size, cfg.pad_to, cfg.dim, cfg.heads,
                                     cfg.ffn_mult, dataset.vocab_size, cfg.lr, device);
    });
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
        case 1: return run_train_select<1>(cfg, device);
        case 2: return run_train_select<2>(cfg, device);
        case 3: return run_train_select<3>(cfg, device);
        case 6: return run_train_select<6>(cfg, device);
        default:
            fmt::print("ERROR: supported layers: 1, 2, 3, 6\n");
            return 1;
    }
}
