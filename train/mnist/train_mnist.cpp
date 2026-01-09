// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MNIST MLP train rebuild (TTNN C++ vs PyTorch functional)

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
#include <memory>
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
    std::string data_dir = "/home/howard/ttnn-perf/experiments/mnist/data";
    std::string outputs_dir = "/home/howard/ttnn-perf/experiments/mnist/outputs";
    uint32_t train_samples = 256;
    uint32_t seed = 1;
    uint32_t pad_to = 1024;
    uint32_t num_classes = 32;
    uint32_t hidden_dim = 256;
    float init_std = 0.02f;
    uint32_t batch_size = 32;
    uint32_t steps = 10;
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float weight_decay = 1e-2f;
    uint32_t dump_steps = 10;
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
    get_u32(kv, "train_samples", cfg.train_samples);
    get_u32(kv, "seed", cfg.seed);
    get_u32(kv, "pad_to", cfg.pad_to);
    get_u32(kv, "num_classes", cfg.num_classes);
    get_u32(kv, "hidden_dim", cfg.hidden_dim);
    get_f32(kv, "init_std", cfg.init_std);
    get_u32(kv, "batch_size", cfg.batch_size);
    get_u32(kv, "steps", cfg.steps);
    get_f32(kv, "lr", cfg.lr);
    get_f32(kv, "beta1", cfg.beta1);
    get_f32(kv, "beta2", cfg.beta2);
    get_f32(kv, "eps", cfg.eps);
    get_f32(kv, "weight_decay", cfg.weight_decay);
    get_u32(kv, "dump_steps", cfg.dump_steps);
    return cfg;
}

std::vector<bfloat16> load_bf16_file(const std::string& path, std::vector<uint32_t>& shape) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open bf16 file: " + path);
    }
    uint32_t ndim = 0;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));
    shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        file.read(reinterpret_cast<char*>(&shape[i]), sizeof(uint32_t));
    }
    size_t numel = 1;
    for (uint32_t d : shape) {
        numel *= d;
    }
    std::vector<bfloat16> data(numel);
    file.read(reinterpret_cast<char*>(data.data()), numel * sizeof(bfloat16));
    return data;
}

std::vector<uint32_t> load_u32_file(const std::string& path, std::vector<uint32_t>& shape) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open u32 file: " + path);
    }
    uint32_t ndim = 0;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));
    shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        file.read(reinterpret_cast<char*>(&shape[i]), sizeof(uint32_t));
    }
    size_t numel = 1;
    for (uint32_t d : shape) {
        numel *= d;
    }
    std::vector<uint32_t> data(numel);
    file.read(reinterpret_cast<char*>(data.data()), numel * sizeof(uint32_t));
    return data;
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

Tensor make_batch_tensor(const std::vector<bfloat16>& batch_data, const ttnn::Shape& shape, MeshDevice& device) {
    auto layout = ttnn::TensorLayout(
        ttnn::DataType::BFLOAT16,
        ttnn::PageConfig(ttnn::ROW_MAJOR_LAYOUT),
        tt::tt_metal::MemoryConfig{}
    );
    auto host = Tensor::from_vector(batch_data, ttnn::TensorSpec(shape, layout));
    return ttnn::tilize(host.to_device(&device));
}

void write_meta(const std::filesystem::path& dir, const Config& cfg) {
    std::filesystem::create_directories(dir);
    std::ofstream file(dir / "meta.json");
    if (!file) {
        return;
    }
    file << "{\n";
    file << "  \"train_samples\": " << cfg.train_samples << ",\n";
    file << "  \"batch_size\": " << cfg.batch_size << ",\n";
    file << "  \"steps\": " << cfg.steps << ",\n";
    file << "  \"pad_to\": " << cfg.pad_to << ",\n";
    file << "  \"num_classes\": " << cfg.num_classes << ",\n";
    file << "  \"hidden_dim\": " << cfg.hidden_dim << ",\n";
    file << "  \"init_std\": " << cfg.init_std << ",\n";
    file << "  \"lr\": " << cfg.lr << ",\n";
    file << "  \"beta1\": " << cfg.beta1 << ",\n";
    file << "  \"beta2\": " << cfg.beta2 << ",\n";
    file << "  \"eps\": " << cfg.eps << ",\n";
    file << "  \"weight_decay\": " << cfg.weight_decay << "\n";
    file << "}\n";
}

struct CrossEntropy2D {
    Tensor one_hot_weight; // [C, C]
    Tensor softmax;        // [B, C]
    Tensor one_hot;        // [B, C]
    Tensor log_probs;      // [B, C]
    Tensor nll;            // [B]
    Tensor total;          // [1]
    Tensor loss;           // [1]
    Tensor d_loss;         // [1]

    uint32_t batch;
    uint32_t classes;
    MeshDevice* device;
    std::vector<uint32_t> index_buf;

    CrossEntropy2D(uint32_t b, uint32_t c, MeshDevice& dev)
        : one_hot_weight(make_zeros(ttnn::Shape({c, c}), dev)),
          softmax(make_zeros(ttnn::Shape({b, c}), dev)),
          one_hot(make_zeros(ttnn::Shape({b, c}), dev)),
          log_probs(make_zeros(ttnn::Shape({b, c}), dev)),
          nll(make_zeros(ttnn::Shape({b}), dev)),
          total(make_zeros(ttnn::Shape({1}), dev)),
          loss(make_zeros(ttnn::Shape({1}), dev)),
          d_loss(make_zeros(ttnn::Shape({1}), dev)),
          batch(b),
          classes(c),
          device(&dev) {
        std::vector<bfloat16> identity(static_cast<size_t>(classes) * classes, bfloat16(0.0f));
        for (uint32_t i = 0; i < classes; ++i) {
            identity[i * classes + i] = bfloat16(1.0f);
        }
        auto tile_layout = ttnn::TensorLayout(
            ttnn::DataType::BFLOAT16,
            ttnn::PageConfig(ttnn::TILE_LAYOUT),
            tt::tt_metal::MemoryConfig{}
        );
        one_hot_weight = Tensor::from_vector(identity, ttnn::TensorSpec(ttnn::Shape({classes, classes}), tile_layout))
                             .to_device(device);
        index_buf.assign(batch, 0);
    }

    Value* build(Graph& g, Value* logits) {
        auto* v = g.node(&loss, &d_loss);
        v->parents = {logits};
        v->backward_fn = [logits, this, v]() {
            if (!v->grad) return;
            auto grad_logits = ttnn::subtract(softmax, one_hot);
            grad_logits = ttnn::multiply(grad_logits, 1.0f / static_cast<float>(batch));
            grad_logits = ttnn::multiply(grad_logits, *v->grad);
            logits->accumulate_grad(grad_logits);
        };
        return v;
    }

    void execute_forward(const Tensor& logits, const std::vector<uint32_t>& labels) {
        if (labels.size() != batch) {
            throw std::runtime_error("labels size mismatch");
        }
        for (uint32_t i = 0; i < batch; ++i) {
            index_buf[i] = labels[i];
        }
        auto indices = make_indices(index_buf, ttnn::Shape({batch}), *device);
        one_hot = ttnn::embedding(indices, one_hot_weight, std::nullopt, ttnn::TILE_LAYOUT);

        softmax = ttnn::softmax(logits, -1, std::nullopt, get_softmax_compute_config(), true);
        log_probs = ttnn::log(softmax);
        auto nll_raw = ttnn::sum(ttnn::multiply(one_hot, log_probs), -1, false);
        nll = ttnn::multiply(nll_raw, -1.0f);
        total = ttnn::sum(nll, 0, false);
        loss = ttnn::multiply(total, 1.0f / static_cast<float>(batch));
    }

    float get_loss() {
        return static_cast<float>(loss.cpu().to_vector<bfloat16>()[0]);
    }
};

struct MnistMLP {
    Linear layer1;
    Linear layer2;
    Tensor relu_out;
    Tensor d_relu_out;
    Tensor relu_mask;

    Tensor input;
    Tensor d_input;

    Graph graph;
    Value* input_node = nullptr;
    Value* logits_node = nullptr;
    Value* loss_node = nullptr;
    bool built = false;

    CrossEntropy2D loss;

    MnistMLP(uint32_t batch, uint32_t in_dim, uint32_t hidden_dim, uint32_t classes, float init, MeshDevice& dev)
        : layer1(batch, in_dim, hidden_dim, init, dev),
          layer2(batch, hidden_dim, classes, init, dev),
          relu_out(make_zeros(ttnn::Shape({batch, hidden_dim}), dev)),
          d_relu_out(make_zeros(ttnn::Shape({batch, hidden_dim}), dev)),
          relu_mask(make_zeros(ttnn::Shape({batch, hidden_dim}), dev)),
          input(make_zeros(ttnn::Shape({batch, in_dim}), dev)),
          d_input(make_zeros(ttnn::Shape({batch, in_dim}), dev)),
          loss(batch, classes, dev) {}

    void build() {
        input_node = graph.leaf(&input, &d_input, false);
        auto* h = layer1.forward(graph, input_node);
        h = relu(graph, h, &relu_out, &d_relu_out, &relu_mask);
        logits_node = layer2.forward(graph, h);
        loss_node = loss.build(graph, logits_node);
        graph.build_topo(loss_node);
        built = true;
    }

    void execute_forward(const Tensor& new_input, const std::vector<uint32_t>& labels) {
        input = new_input;
        layer1.mm_out = ttnn::matmul(input, layer1.weight, false, true);
        layer1.out = ttnn::add(layer1.mm_out, layer1.bias);
        relu_mask = ttnn::gtz(layer1.out);
        relu_out = ttnn::relu(layer1.out);
        layer2.mm_out = ttnn::matmul(relu_out, layer2.weight, false, true);
        layer2.out = ttnn::add(layer2.mm_out, layer2.bias);
        loss.execute_forward(layer2.out, labels);
    }
};

void dump_step(const std::filesystem::path& step_dir,
               MnistMLP& model,
               const std::vector<uint32_t>& labels,
               MeshDevice& device) {
    std::filesystem::create_directories(step_dir);
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    traced::save_tensor(model.input, (step_dir / "input.bin").string(), &device);
    save_u32_tensor(labels, ttnn::Shape({static_cast<uint32_t>(labels.size())}), (step_dir / "labels_u32.bin").string());

    traced::save_tensor(model.layer1.weight, (step_dir / "w1.bin").string(), &device);
    traced::save_tensor(model.layer1.bias, (step_dir / "b1.bin").string(), &device);
    traced::save_tensor(model.layer2.weight, (step_dir / "w2.bin").string(), &device);
    traced::save_tensor(model.layer2.bias, (step_dir / "b2.bin").string(), &device);

    traced::save_tensor(model.layer1.out, (step_dir / "linear1_out.bin").string(), &device);
    traced::save_tensor(model.relu_out, (step_dir / "relu_out.bin").string(), &device);
    traced::save_tensor(model.layer2.out, (step_dir / "logits.bin").string(), &device);
    traced::save_tensor(model.loss.softmax, (step_dir / "softmax.bin").string(), &device);
    traced::save_tensor(model.loss.loss, (step_dir / "loss.bin").string(), &device);

    traced::save_tensor(model.layer1.d_weight, (step_dir / "w1_grad.bin").string(), &device);
    traced::save_tensor(model.layer1.d_bias, (step_dir / "b1_grad.bin").string(), &device);
    traced::save_tensor(model.layer2.d_weight, (step_dir / "w2_grad.bin").string(), &device);
    traced::save_tensor(model.layer2.d_bias, (step_dir / "b2_grad.bin").string(), &device);
    traced::save_tensor(model.layer2.d_out, (step_dir / "logits_grad.bin").string(), &device);
}

} // namespace

int main() {
    const char* env_path = std::getenv("TRAIN_MNIST_CONFIG");
    std::string config_path = env_path
        ? env_path
        : "/home/howard/ttnn-perf/train/mnist/default.yaml";

    Config cfg = load_config(config_path);

    fmt::print("# MNIST config: {}\n", config_path);
    fmt::print("# batch={}, steps={}, pad_to={}, hidden={}, classes={}\n",
               cfg.batch_size, cfg.steps, cfg.pad_to, cfg.hidden_dim, cfg.num_classes);

    if (cfg.batch_size % 32 != 0 || cfg.pad_to % 32 != 0 || cfg.hidden_dim % 32 != 0 || cfg.num_classes % 32 != 0) {
        fmt::print("ERROR: batch_size, pad_to, hidden_dim, num_classes must be multiples of 32\n");
        return 1;
    }

    DeviceGuard guard;
    MeshDevice& device = guard.get();

    g_seed_counter = std::max<uint32_t>(1, cfg.seed);

    auto images_path = std::filesystem::path(cfg.data_dir) / "train_images.bin";
    auto labels_path = std::filesystem::path(cfg.data_dir) / "train_labels_u32.bin";

    std::vector<uint32_t> img_shape;
    std::vector<uint32_t> lbl_shape;
    auto images = load_bf16_file(images_path.string(), img_shape);
    auto labels = load_u32_file(labels_path.string(), lbl_shape);

    if (img_shape.size() != 2 || img_shape[0] != cfg.train_samples || img_shape[1] != cfg.pad_to) {
        fmt::print("ERROR: images shape mismatch (expected [{}x{}])\n", cfg.train_samples, cfg.pad_to);
        return 1;
    }
    if (lbl_shape.size() != 1 || lbl_shape[0] != cfg.train_samples) {
        fmt::print("ERROR: labels shape mismatch (expected [{}])\n", cfg.train_samples);
        return 1;
    }

    MnistMLP model(cfg.batch_size, cfg.pad_to, cfg.hidden_dim, cfg.num_classes, cfg.init_std, device);
    Adam adam(cfg.lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);
    adam.add_param(&model.layer1.weight, &model.layer1.d_weight, device, true);
    adam.add_param(&model.layer1.bias, &model.layer1.d_bias, device, false);
    adam.add_param(&model.layer2.weight, &model.layer2.d_weight, device, true);
    adam.add_param(&model.layer2.bias, &model.layer2.d_bias, device, false);

    model.build();

    auto outputs_dir = std::filesystem::path(cfg.outputs_dir) / "cpp";
    write_meta(outputs_dir, cfg);

    auto losses_path = outputs_dir / "losses.tsv";
    {
        std::ofstream file(losses_path);
        file << "step\tloss\n";
    }

    std::vector<bfloat16> batch_data(static_cast<size_t>(cfg.batch_size) * cfg.pad_to);
    std::vector<uint32_t> batch_labels(cfg.batch_size);

    for (uint32_t step = 0; step < cfg.steps; ++step) {
        uint32_t start = (step * cfg.batch_size) % cfg.train_samples;
        for (uint32_t i = 0; i < cfg.batch_size; ++i) {
            uint32_t idx = (start + i) % cfg.train_samples;
            const size_t src_offset = static_cast<size_t>(idx) * cfg.pad_to;
            const size_t dst_offset = static_cast<size_t>(i) * cfg.pad_to;
            std::copy(images.begin() + src_offset, images.begin() + src_offset + cfg.pad_to,
                      batch_data.begin() + dst_offset);
            batch_labels[i] = labels[idx];
            if (batch_labels[i] >= cfg.num_classes) {
                fmt::print("ERROR: label {} >= num_classes {}\n", batch_labels[i], cfg.num_classes);
                return 1;
            }
        }

        Tensor input = make_batch_tensor(batch_data, ttnn::Shape({cfg.batch_size, cfg.pad_to}), device);

        {
#ifdef TRACY_ENABLE
            ZoneScopedN("forward");
#endif
            model.execute_forward(input, batch_labels);
            tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        }

        {
#ifdef TRACY_ENABLE
            ZoneScopedN("backward");
#endif
            model.graph.zero_grad();
            model.graph.backward_scaled(model.loss_node, 1.0f);
            tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        }

        if (step < cfg.dump_steps) {
            auto step_dir = outputs_dir / fmt::format("step_{:04d}", step);
            dump_step(step_dir, model, batch_labels, device);
        }

        float loss_val = model.loss.get_loss();
        {
            std::ofstream file(losses_path, std::ios::app);
            file << step << "\t" << loss_val << "\n";
        }
        fmt::print("step {} | loss {:.6f}\n", step, loss_val);

        {
#ifdef TRACY_ENABLE
            ZoneScopedN("adam_step");
#endif
            adam.step();
            tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        }
    }

    return 0;
}
