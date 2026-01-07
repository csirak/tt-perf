// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Attention forward pass with per-layer timing.
// Measures actual TFLOPS for each operation.
//
// Usage:
//   make verify-attention-timed

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
#include <chrono>
#include <vector>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using Tensor = tt::tt_metal::Tensor;

// Configuration
constexpr uint32_t BATCH = 32;
constexpr uint32_t SEQ = 64;
constexpr uint32_t DIM = 256;
constexpr uint32_t HEADS = 8;
constexpr uint32_t HEAD_DIM = DIM / HEADS;
constexpr int WARMUP_ITERS = 3;
constexpr int TIMED_ITERS = 10;

const std::string INPUT_DIR = "verify/outputs/pytorch";

// FLOPs calculations
constexpr double QKV_PROJ_FLOPS = 2.0 * BATCH * SEQ * DIM * DIM;  // matmul
constexpr double SCORES_FLOPS = 2.0 * BATCH * HEADS * SEQ * SEQ * HEAD_DIM;  // Q @ K.T
constexpr double SCALE_FLOPS = BATCH * HEADS * SEQ * SEQ;  // elementwise
constexpr double MASK_FLOPS = BATCH * HEADS * SEQ * SEQ;  // elementwise
constexpr double SOFTMAX_FLOPS = 5.0 * BATCH * HEADS * SEQ * SEQ;  // exp/sum/div
constexpr double ATTN_OUT_FLOPS = 2.0 * BATCH * HEADS * SEQ * HEAD_DIM * SEQ;  // W @ V
constexpr double OUTPUT_FLOPS = 2.0 * BATCH * SEQ * DIM * DIM;  // output proj

struct LayerTiming {
    std::string name;
    double flops;
    double total_us;
    int count;

    double avg_us() const { return total_us / count; }
    double tflops() const { return (flops / avg_us()) / 1e6; }  // TFLOPS
};

int main() {
    fmt::print("======================================================================\n");
    fmt::print("C++ Attention Forward - Per-Layer Timing\n");
    fmt::print("======================================================================\n");
    fmt::print("Config: batch={}, seq={}, dim={}, heads={}\n", BATCH, SEQ, DIM, HEADS);
    fmt::print("Warmup: {} iters, Timed: {} iters\n\n", WARMUP_ITERS, TIMED_ITERS);

    // Open device
    auto device = MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 64 * 1024 * 1024);
    auto grid = device->compute_with_storage_grid_size();
    fmt::print("Compute grid: {}x{} = {} cores\n\n", grid.x, grid.y, grid.x * grid.y);

    // Timing accumulators
    std::vector<LayerTiming> timings = {
        {"q_proj", QKV_PROJ_FLOPS, 0, 0},
        {"k_proj", QKV_PROJ_FLOPS, 0, 0},
        {"v_proj", QKV_PROJ_FLOPS, 0, 0},
        {"reshape_qkv", 0, 0, 0},
        {"scores", SCORES_FLOPS + SCALE_FLOPS, 0, 0},
        {"mask", MASK_FLOPS, 0, 0},
        {"softmax", SOFTMAX_FLOPS, 0, 0},
        {"attn_out", ATTN_OUT_FLOPS, 0, 0},
        {"merge", 0, 0, 0},
        {"output", OUTPUT_FLOPS, 0, 0},
    };

    auto sync = [&]() {
        tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
    };

    auto time_op = [&](int idx, auto&& op) {
        sync();
        auto start = std::chrono::high_resolution_clock::now();
        auto result = op();
        sync();
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(end - start).count();
        timings[idx].total_us += us;
        timings[idx].count++;
        return result;
    };

    try {
        // Load weights and input
        fmt::print("Loading weights...\n");
        auto wq = traced::load_tensor(INPUT_DIR + "/wq.bin", *device);
        auto wk = traced::load_tensor(INPUT_DIR + "/wk.bin", *device);
        auto wv = traced::load_tensor(INPUT_DIR + "/wv.bin", *device);
        auto wo = traced::load_tensor(INPUT_DIR + "/wo.bin", *device);
        auto x = traced::load_tensor(INPUT_DIR + "/input.bin", *device);
        auto mask = traced::load_tensor(INPUT_DIR + "/causal_mask.bin", *device);
        sync();

        fmt::print("Running {} warmup iterations...\n", WARMUP_ITERS);
        for (int i = 0; i < WARMUP_ITERS; ++i) {
            auto q_proj = ttnn::matmul(x, wq, false, true);
            auto k_proj = ttnn::matmul(x, wk, false, true);
            auto v_proj = ttnn::matmul(x, wv, false, true);
            auto q = ttnn::transpose(ttnn::reshape(q_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
            auto k = ttnn::transpose(ttnn::reshape(k_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
            auto v = ttnn::transpose(ttnn::reshape(v_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
            float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
            auto scores = ttnn::multiply(ttnn::matmul(q, k, false, true), scale);
            auto scores_masked = ttnn::add(scores, mask);
            auto attn_weights = ttnn::softmax(scores_masked, -1);
            auto attn_out = ttnn::matmul(attn_weights, v);
            auto attn_merged = ttnn::reshape(ttnn::transpose(attn_out, 1, 2), ttnn::Shape({BATCH, SEQ, DIM}));
            auto output = ttnn::matmul(attn_merged, wo, false, true);
            sync();
        }

        fmt::print("Running {} timed iterations...\n\n", TIMED_ITERS);
        for (int iter = 0; iter < TIMED_ITERS; ++iter) {
            // QKV projections
            auto q_proj = time_op(0, [&]() { return ttnn::matmul(x, wq, false, true); });
            auto k_proj = time_op(1, [&]() { return ttnn::matmul(x, wk, false, true); });
            auto v_proj = time_op(2, [&]() { return ttnn::matmul(x, wv, false, true); });

            // Reshape
            Tensor q, k, v;
            time_op(3, [&]() {
                q = ttnn::transpose(ttnn::reshape(q_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
                k = ttnn::transpose(ttnn::reshape(k_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
                v = ttnn::transpose(ttnn::reshape(v_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
                return 0;
            });

            // Scores
            float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));
            auto scores = time_op(4, [&]() {
                return ttnn::multiply(ttnn::matmul(q, k, false, true), scale);
            });

            // Mask
            auto scores_masked = time_op(5, [&]() { return ttnn::add(scores, mask); });

            // Softmax
            auto attn_weights = time_op(6, [&]() { return ttnn::softmax(scores_masked, -1); });

            // Attention output
            auto attn_out = time_op(7, [&]() { return ttnn::matmul(attn_weights, v); });

            // Merge heads
            Tensor attn_merged;
            time_op(8, [&]() {
                attn_merged = ttnn::reshape(ttnn::transpose(attn_out, 1, 2), ttnn::Shape({BATCH, SEQ, DIM}));
                return 0;
            });

            // Output projection
            auto output = time_op(9, [&]() { return ttnn::matmul(attn_merged, wo, false, true); });
        }

        // Print results
        fmt::print("==========================================================================================\n");
        fmt::print("{:<15} {:>12} {:>12} {:>12} {:>12}\n",
                   "Layer", "MFLOPs", "Avg (us)", "TFLOPS", "% of total");
        fmt::print("------------------------------------------------------------------------------------------\n");

        double total_us = 0;
        double total_flops = 0;
        for (const auto& t : timings) {
            total_us += t.avg_us();
            total_flops += t.flops;
        }

        for (const auto& t : timings) {
            double mflops = t.flops / 1e6;
            double pct = (t.avg_us() / total_us) * 100;
            if (t.flops > 0) {
                fmt::print("{:<15} {:>12.2f} {:>12.2f} {:>12.4f} {:>11.1f}%\n",
                           t.name, mflops, t.avg_us(), t.tflops(), pct);
            } else {
                fmt::print("{:<15} {:>12} {:>12.2f} {:>12} {:>11.1f}%\n",
                           t.name, "-", t.avg_us(), "-", pct);
            }
        }

        fmt::print("------------------------------------------------------------------------------------------\n");
        double total_tflops = (total_flops / total_us) / 1e6;
        fmt::print("{:<15} {:>12.2f} {:>12.2f} {:>12.4f} {:>11}%\n",
                   "TOTAL", total_flops / 1e6, total_us, total_tflops, "100.0");

    } catch (const std::exception& e) {
        fmt::print("ERROR: {}\n", e.what());
        tt::tt_metal::distributed::Finish(device->mesh_command_queue());
        ttnn::close_device(*device);
        return 1;
    }

    tt::tt_metal::distributed::Finish(device->mesh_command_queue());
    ttnn::close_device(*device);
    fmt::print("\nDone.\n");
    return 0;
}
