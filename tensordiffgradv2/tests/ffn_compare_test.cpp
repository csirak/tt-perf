// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// FFN comparison test: FP32 Torch vs BF16 Torch vs BF16 TTNN
// Uses SHARED weight initialization via TDF1 file serialization.
//
// Build modes:
// - Torch-only: cmake build, runs FP32 + BF16 Torch, saves weights to TDF1
// - TTNN: tt-metal build with -DFFN_TTNN_MODE, loads weights from TDF1

// Check for TTNN mode (set by tt-metal CMakeLists.txt)
#ifdef FFN_TTNN_MODE
#define HAS_TTNN 1
// TTNN build - needs ttnn headers, include paths relative to tensordiffgradv2/
#include "backends/ttnn_tensor.hpp"
#include "autograd/graph.hpp"
#include "ops/reglu.hpp"
#include <ttnn/device.hpp>
#else
#define HAS_TTNN 0
// Torch-only build - include paths relative to parent dir
#include "tensordiffgradv2/backends/torch_tensor.hpp"
#include "tensordiffgradv2/autograd/graph.hpp"
#include "tensordiffgradv2/ops/reglu.hpp"
#endif

#include <iostream>
#include <string>
#include <filesystem>

using namespace tensordiffgradv2;

// Tile-aligned dimensions
constexpr int64_t B = 1;
constexpr int64_t S = 32;
constexpr int64_t D = 64;
constexpr int64_t F = 256;

// Weight file paths
const std::string WEIGHT_DIR = "outputs/weights";

#if !HAS_TTNN
// ========== TORCH-ONLY MODE ==========
// Runs FP32 + BF16 Torch and saves weights to TDF1 files

// Shared FP32 weight storage for ReGLU FFN
// ReGLU: gate = ReLU(x @ W_gate + b_gate), value = x @ W_value + b_value
//        hidden = gate * value, output = hidden @ W2 + b2
torch::Tensor x_fp32, w_gate_fp32, b_gate_fp32, w_value_fp32, b_value_fp32, w2_fp32, b2_fp32;

void init_and_save_weights() {
    torch::manual_seed(42);
    // Xavier initialization: scale by 1/sqrt(fan_in)
    x_fp32 = torch::randn({B, S, D}, torch::kFloat32) / std::sqrt((float)D);
    w_gate_fp32 = torch::randn({D, F}, torch::kFloat32) / std::sqrt((float)D);
    b_gate_fp32 = torch::zeros({1, 1, F}, torch::kFloat32);
    w_value_fp32 = torch::randn({D, F}, torch::kFloat32) / std::sqrt((float)D);
    b_value_fp32 = torch::zeros({1, 1, F}, torch::kFloat32);
    w2_fp32 = torch::randn({F, D}, torch::kFloat32) / std::sqrt((float)F);
    b2_fp32 = torch::zeros({1, 1, D}, torch::kFloat32);

    std::cout << "Initialized ReGLU FFN weights in FP32\n";
    std::cout << "  x: [" << B << "," << S << "," << D << "]\n";
    std::cout << "  w_gate: [" << D << "," << F << "]\n";
    std::cout << "  b_gate: [1,1," << F << "]\n";
    std::cout << "  w_value: [" << D << "," << F << "]\n";
    std::cout << "  b_value: [1,1," << F << "]\n";
    std::cout << "  w2: [" << F << "," << D << "]\n";
    std::cout << "  b2: [1,1," << D << "]\n";

    // Save weights as TDF1 files for TTNN to load
    std::filesystem::create_directories(WEIGHT_DIR);

    // Convert to BF16 and save (TTNN needs BF16)
    TorchTensor(x_fp32.to(torch::kBFloat16)).save(WEIGHT_DIR + "/x.bin");
    TorchTensor(w_gate_fp32.to(torch::kBFloat16)).save(WEIGHT_DIR + "/w_gate.bin");
    TorchTensor(b_gate_fp32.to(torch::kBFloat16)).save(WEIGHT_DIR + "/b_gate.bin");
    TorchTensor(w_value_fp32.to(torch::kBFloat16)).save(WEIGHT_DIR + "/w_value.bin");
    TorchTensor(b_value_fp32.to(torch::kBFloat16)).save(WEIGHT_DIR + "/b_value.bin");
    TorchTensor(w2_fp32.to(torch::kBFloat16)).save(WEIGHT_DIR + "/w2.bin");
    TorchTensor(b2_fp32.to(torch::kBFloat16)).save(WEIGHT_DIR + "/b2.bin");

    std::cout << "\nSaved weights to " << WEIGHT_DIR << "/ for TTNN test\n";
}

void run_torch(const std::string& out_dir, torch::Dtype torch_dtype) {
    std::string dtype_name = (torch_dtype == torch::kFloat32) ? "fp32" : "bf16";
    std::cout << "\n=== Running Torch " << dtype_name << " ===\n";
    std::cout << "  output: " << out_dir << "\n";

    Graph graph;
    graph.enable_logging(out_dir);

    // Convert FP32 weights to target dtype and wrap
    auto x = std::make_shared<TorchTensor>(x_fp32.to(torch_dtype));
    auto w_gate = std::make_shared<TorchTensor>(w_gate_fp32.to(torch_dtype));
    auto b_gate = std::make_shared<TorchTensor>(b_gate_fp32.to(torch_dtype));
    auto w_value = std::make_shared<TorchTensor>(w_value_fp32.to(torch_dtype));
    auto b_value = std::make_shared<TorchTensor>(b_value_fp32.to(torch_dtype));
    auto w2 = std::make_shared<TorchTensor>(w2_fp32.to(torch_dtype));
    auto b2 = std::make_shared<TorchTensor>(b2_fp32.to(torch_dtype));

    // Wrap in Graph
    auto* vx = graph.input(x, "act");
    auto* vw_gate = graph.param(w_gate, "w_gate");
    auto* vb_gate = graph.param(b_gate, "b_gate");
    auto* vw_value = graph.param(w_value, "w_value");
    auto* vb_value = graph.param(b_value, "b_value");
    auto* vw2 = graph.param(w2, "w2");
    auto* vb2 = graph.param(b2, "b2");

    // ReGLU forward using composite op
    auto* h = ops::reglu_forward(graph, vx,
        vw_gate, vb_gate,
        vw_value, vb_value,
        vw2, vb2,
        "reglu");

    auto* loss = h->sum("loss");
    (void)loss;

    std::cout << "  Forward pass complete.\n";
    std::cout << "  Output shape: [" << h->data->shape()[0] << ","
              << h->data->shape()[1] << "," << h->data->shape()[2] << "]\n";
}
#endif  // !HAS_TTNN

#if HAS_TTNN
// ========== TTNN MODE ==========
// Loads weights from TDF1 files and runs TTNN

// Load a TDF1 file into TTNN tensor
std::shared_ptr<TtnnTensor> load_tdf1_to_ttnn(const std::string& path, MeshDevice& device) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open TDF1 file: " + path);
    }

    // Read TDF1 header
    char magic[4];
    uint32_t version, dtype_code, ndim;
    file.read(magic, 4);
    if (std::strncmp(magic, "TDF1", 4) != 0) {
        throw std::runtime_error("Invalid TDF1 magic: " + path);
    }
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&dtype_code), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));

    // Read shape
    std::vector<uint32_t> dims(ndim);
    std::vector<int64_t> shape(ndim);
    size_t numel = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        file.read(reinterpret_cast<char*>(&dims[i]), sizeof(uint32_t));
        shape[i] = dims[i];
        numel *= dims[i];
    }

    // Read data (assume BF16)
    std::vector<bfloat16> data(numel);
    file.read(reinterpret_cast<char*>(data.data()), numel * sizeof(uint16_t));

    // Create TTNN tensor
    ttnn::Shape tt_shape(dims);
    auto layout = ttnn::TensorLayout(
        ttnn::DataType::BFLOAT16,
        ttnn::PageConfig(ttnn::TILE_LAYOUT),
        tt::tt_metal::MemoryConfig{}
    );
    auto tensor = tt::tt_metal::Tensor::from_vector(data, ttnn::TensorSpec(tt_shape, layout));
    return std::make_shared<TtnnTensor>(tensor.to_device(&device), &device);
}

void run_ttnn(const std::string& out_dir, MeshDevice& device) {
    std::cout << "\n=== Running TTNN BF16 ===\n";
    std::cout << "  output: " << out_dir << "\n";
    std::cout << "  Loading weights from " << WEIGHT_DIR << "/\n";

    // Load weights from TDF1 files
    auto x = load_tdf1_to_ttnn(WEIGHT_DIR + "/x.bin", device);
    auto w_gate = load_tdf1_to_ttnn(WEIGHT_DIR + "/w_gate.bin", device);
    auto b_gate = load_tdf1_to_ttnn(WEIGHT_DIR + "/b_gate.bin", device);
    auto w_value = load_tdf1_to_ttnn(WEIGHT_DIR + "/w_value.bin", device);
    auto b_value = load_tdf1_to_ttnn(WEIGHT_DIR + "/b_value.bin", device);
    auto w2 = load_tdf1_to_ttnn(WEIGHT_DIR + "/w2.bin", device);
    auto b2 = load_tdf1_to_ttnn(WEIGHT_DIR + "/b2.bin", device);

    Graph graph;
    graph.enable_logging(out_dir);

    auto* vx = graph.input(x, "act");
    auto* vw_gate = graph.param(w_gate, "w_gate");
    auto* vb_gate = graph.param(b_gate, "b_gate");
    auto* vw_value = graph.param(w_value, "w_value");
    auto* vb_value = graph.param(b_value, "b_value");
    auto* vw2 = graph.param(w2, "w2");
    auto* vb2 = graph.param(b2, "b2");

    // ReGLU forward using composite op
    auto* h = ops::reglu_forward(graph, vx,
        vw_gate, vb_gate,
        vw_value, vb_value,
        vw2, vb2,
        "reglu");

    auto* loss = h->sum("loss");
    (void)loss;

    std::cout << "  Forward pass complete.\n";
    auto s = h->data->shape();
    std::cout << "  Output shape: [" << s[0] << "," << s[1] << "," << s[2] << "]\n";
}
#endif  // HAS_TTNN

int main() {
    std::cout << "FFN Comparison Test\n";
    std::cout << "==================\n";
    std::cout << "Shape: B=" << B << " S=" << S << " D=" << D << " F=" << F << "\n";

#if !HAS_TTNN
    // Torch-only mode: run FP32 and BF16 Torch
    init_and_save_weights();

    // Run Torch backends (all ops logged with hierarchy)
    run_torch("outputs/ffn_fp32", torch::kFloat32);
    run_torch("outputs/ffn_bf16_torch", torch::kBFloat16);

    std::cout << "\n=== Complete ===\n";
    std::cout << "Torch backends (FP32 + BF16) done.\n";
    std::cout << "Weights saved to " << WEIGHT_DIR << "/\n";
    std::cout << "Run TTNN on howard to complete comparison.\n";
#else
    // TTNN mode: run BF16 TTNN
    std::cout << "\nOpening device...\n";
    auto device = ttnn::open_mesh_device(0);

    run_ttnn("outputs/ffn_bf16_ttnn", *device);

    std::cout << "\nClosing device...\n";
    ttnn::close_device(*device);

    std::cout << "\n=== Complete ===\n";
    std::cout << "TTNN backend done.\n";
#endif  // HAS_TTNN

    std::cout << "\nView comparison:\n";
    std::cout << "  cd viewer && COMPARE_DIR0=../outputs/ffn_fp32 COMPARE_DIR1=../outputs/ffn_bf16_ttnn yarn start\n";

    return 0;
}
