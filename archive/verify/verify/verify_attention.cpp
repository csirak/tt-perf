// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Attention forward pass verification using our traced autograd.
// Loads weights from PyTorch-generated files, runs forward, saves outputs.
//
// Usage:
//   make verify-attention-build
//   make verify-attention-run

#include "../autograd/traced/tensor_io.hpp"
#include "../autograd/traced/ops.hpp"
#include "../autograd/traced/nn.hpp"
#include "../autograd/common.hpp"

#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>
#include <filesystem>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using Tensor = tt::tt_metal::Tensor;

// Configuration - must match Python gen_pytorch_outputs.py
constexpr uint32_t BATCH = 32;
constexpr uint32_t SEQ = 64;
constexpr uint32_t DIM = 256;
constexpr uint32_t HEADS = 8;
constexpr uint32_t HEAD_DIM = DIM / HEADS;

const std::string INPUT_DIR = "verify/outputs/pytorch";
const std::string OUTPUT_DIR = "verify/outputs/cpp";

int main() {
    fmt::print("======================================================================\n");
    fmt::print("C++ Attention Forward Verification\n");
    fmt::print("======================================================================\n");
    fmt::print("Config: batch={}, seq={}, dim={}, heads={}\n", BATCH, SEQ, DIM, HEADS);
    fmt::print("Input dir: {}\n", INPUT_DIR);
    fmt::print("Output dir: {}\n\n", OUTPUT_DIR);

    // Create output directory
    std::filesystem::create_directories(OUTPUT_DIR);

    // Open device
    fmt::print("Opening device...\n");
    auto device = MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 64 * 1024 * 1024);
    auto grid = device->compute_with_storage_grid_size();
    fmt::print("Compute grid: {}x{} = {} cores\n\n", grid.x, grid.y, grid.x * grid.y);

    try {
        // Load weights
        fmt::print("Loading weights from {}...\n", INPUT_DIR);
        auto wq = traced::load_tensor(INPUT_DIR + "/wq.bin", *device);
        auto wk = traced::load_tensor(INPUT_DIR + "/wk.bin", *device);
        auto wv = traced::load_tensor(INPUT_DIR + "/wv.bin", *device);
        auto wo = traced::load_tensor(INPUT_DIR + "/wo.bin", *device);
        fmt::print("  wq, wk, wv, wo: [{}, {}]\n", DIM, DIM);

        // Load input
        auto x = traced::load_tensor(INPUT_DIR + "/input.bin", *device);
        fmt::print("  input: [{}, {}, {}]\n", BATCH, SEQ, DIM);

        // Load causal mask
        auto mask = traced::load_tensor(INPUT_DIR + "/causal_mask.bin", *device);
        fmt::print("  causal_mask: [1, 1, {}, {}]\n", SEQ, SEQ);

        // Forward pass
        fmt::print("\nRunning forward pass...\n");

        // QKV projections: y = x @ W.T
        auto q_proj = ttnn::matmul(x, wq, false, true);
        auto k_proj = ttnn::matmul(x, wk, false, true);
        auto v_proj = ttnn::matmul(x, wv, false, true);

        traced::save_tensor(q_proj, OUTPUT_DIR + "/q_proj.bin", device.get());
        traced::save_tensor(k_proj, OUTPUT_DIR + "/k_proj.bin", device.get());
        traced::save_tensor(v_proj, OUTPUT_DIR + "/v_proj.bin", device.get());
        fmt::print("  q_proj, k_proj, v_proj: [{}, {}, {}]\n", BATCH, SEQ, DIM);

        // Reshape for multi-head: [B, S, D] -> [B, S, H, D/H] -> [B, H, S, D/H]
        auto q = ttnn::reshape(q_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM}));
        q = ttnn::transpose(q, 1, 2);
        auto k = ttnn::reshape(k_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM}));
        k = ttnn::transpose(k, 1, 2);
        auto v = ttnn::reshape(v_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM}));
        v = ttnn::transpose(v, 1, 2);

        traced::save_tensor(q, OUTPUT_DIR + "/q.bin", device.get());
        traced::save_tensor(k, OUTPUT_DIR + "/k.bin", device.get());
        traced::save_tensor(v, OUTPUT_DIR + "/v.bin", device.get());
        fmt::print("  q, k, v (reshaped): [{}, {}, {}, {}]\n", BATCH, HEADS, SEQ, HEAD_DIM);

        // Attention scores: Q @ K.T * scale
        float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
        auto scores = ttnn::matmul(q, k, false, true);
        scores = ttnn::multiply(scores, scale);

        traced::save_tensor(scores, OUTPUT_DIR + "/scores.bin", device.get());
        fmt::print("  scores: [{}, {}, {}, {}]\n", BATCH, HEADS, SEQ, SEQ);

        // Apply causal mask
        auto scores_masked = ttnn::add(scores, mask);

        traced::save_tensor(scores_masked, OUTPUT_DIR + "/scores_masked.bin", device.get());
        fmt::print("  scores_masked: [{}, {}, {}, {}]\n", BATCH, HEADS, SEQ, SEQ);

        // Softmax
        auto attn_weights = ttnn::softmax(scores_masked, -1);

        traced::save_tensor(attn_weights, OUTPUT_DIR + "/attn_weights.bin", device.get());
        fmt::print("  attn_weights: [{}, {}, {}, {}]\n", BATCH, HEADS, SEQ, SEQ);

        // Attention output: weights @ V
        auto attn_out = ttnn::matmul(attn_weights, v);

        traced::save_tensor(attn_out, OUTPUT_DIR + "/attn_out.bin", device.get());
        fmt::print("  attn_out: [{}, {}, {}, {}]\n", BATCH, HEADS, SEQ, HEAD_DIM);

        // Merge heads: [B, H, S, D/H] -> [B, S, H, D/H] -> [B, S, D]
        auto attn_merged = ttnn::transpose(attn_out, 1, 2);
        attn_merged = ttnn::reshape(attn_merged, ttnn::Shape({BATCH, SEQ, DIM}));

        traced::save_tensor(attn_merged, OUTPUT_DIR + "/attn_merged.bin", device.get());
        fmt::print("  attn_merged: [{}, {}, {}]\n", BATCH, SEQ, DIM);

        // Output projection
        auto output = ttnn::matmul(attn_merged, wo, false, true);

        traced::save_tensor(output, OUTPUT_DIR + "/output.bin", device.get());
        fmt::print("  output: [{}, {}, {}]\n", BATCH, SEQ, DIM);

        // Summary
        tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
        auto output_vec = output.cpu().to_vector<bfloat16>();
        float sum = 0.0f;
        for (size_t i = 0; i < std::min(size_t(100), output_vec.size()); ++i) {
            sum += static_cast<float>(output_vec[i]);
        }
        fmt::print("\nOutput first 100 elements sum: {:.6f}\n", sum);

        // Count files
        int file_count = 0;
        for (const auto& entry : std::filesystem::directory_iterator(OUTPUT_DIR)) {
            if (entry.path().extension() == ".bin") file_count++;
        }
        fmt::print("Saved {} files to {}\n", file_count, OUTPUT_DIR);

    } catch (const std::exception& e) {
        fmt::print("ERROR: {}\n", e.what());
        tt::tt_metal::distributed::Finish(device->mesh_command_queue());
        ttnn::close_device(*device);
        return 1;
    }

    // Cleanup
    tt::tt_metal::distributed::Finish(device->mesh_command_queue());
    ttnn::close_device(*device);

    fmt::print("\nDone.\n");
    return 0;
}
