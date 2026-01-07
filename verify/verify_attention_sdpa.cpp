// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Attention forward pass using TTNN fused SDPA vs manual implementation.
// Compares timing and accuracy between the two approaches.
//
// Usage:
//   make verify-attention-sdpa

#include "../autograd/traced/tensor_io.hpp"
#include "../autograd/traced/ops.hpp"
#include "../autograd/traced/nn.hpp"
#include "../autograd/common.hpp"

#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>
#include <ttnn/operations/transformer/sdpa/sdpa.hpp>
#include <ttnn/operations/transformer/sdpa_config.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <fmt/core.h>
#include <filesystem>
#include <chrono>
#include <vector>
#include <cmath>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using Tensor = tt::tt_metal::Tensor;

// Configuration - must match verify_attention_timed.cpp
constexpr uint32_t BATCH = 32;
constexpr uint32_t SEQ = 64;
constexpr uint32_t DIM = 512;
constexpr uint32_t HEADS = 8;
constexpr uint32_t HEAD_DIM = DIM / HEADS;
constexpr int WARMUP_ITERS = 3;
constexpr int TIMED_ITERS = 10;

const std::string INPUT_DIR = "verify/outputs/pytorch";
const std::string OUTPUT_DIR = "verify/outputs/cpp_sdpa";

// Attention FLOPs calculated from config:
// Q@K.T: 2 * B * H * S * S * head_dim
// Scale: B * H * S * S
// Mask:  B * H * S * S
// Softmax: 5 * B * H * S * S
// W@V:   2 * B * H * S * head_dim * S
constexpr double ATTENTION_CORE_FLOPS =
    2.0 * BATCH * HEADS * SEQ * SEQ * HEAD_DIM +  // Q@K.T
    BATCH * HEADS * SEQ * SEQ +                    // Scale
    BATCH * HEADS * SEQ * SEQ +                    // Mask
    5.0 * BATCH * HEADS * SEQ * SEQ +              // Softmax
    2.0 * BATCH * HEADS * SEQ * HEAD_DIM * SEQ;    // W@V

// Full attention with projections:
// QKV projections: 3 * 2 * B * S * D * D
// Output projection: 2 * B * S * D * D
constexpr double QKV_PROJ_FLOPS = 3.0 * 2.0 * BATCH * SEQ * DIM * DIM;
constexpr double OUT_PROJ_FLOPS = 2.0 * BATCH * SEQ * DIM * DIM;
constexpr double FULL_ATTENTION_FLOPS = QKV_PROJ_FLOPS + ATTENTION_CORE_FLOPS + OUT_PROJ_FLOPS;

int main() {
    fmt::print("======================================================================\n");
    fmt::print("SDPA vs Manual Attention Comparison\n");
    fmt::print("======================================================================\n");
    fmt::print("Config: batch={}, seq={}, dim={}, heads={}, head_dim={}\n",
               BATCH, SEQ, DIM, HEADS, HEAD_DIM);
    fmt::print("Warmup: {} iters, Timed: {} iters\n", WARMUP_ITERS, TIMED_ITERS);
    fmt::print("Attention core FLOPs: {:.2f} MFLOPs\n", ATTENTION_CORE_FLOPS / 1e6);
    fmt::print("Full attention FLOPs: {:.2f} MFLOPs\n\n", FULL_ATTENTION_FLOPS / 1e6);

    // Create output directory
    std::filesystem::create_directories(OUTPUT_DIR);

    // Open device
    auto device = MeshDevice::create_unit_mesh(0, DEFAULT_L1_SMALL_SIZE, 64 * 1024 * 1024);
    auto grid = device->compute_with_storage_grid_size();
    fmt::print("Compute grid: {}x{} = {} cores\n\n", grid.x, grid.y, grid.x * grid.y);

    auto sync = [&]() {
        tt::tt_metal::distributed::Synchronize(device.get(), std::nullopt);
    };

    // Compute config for matmul ops
    auto compute_config = ttnn::WormholeComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi2,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = true,
    };

    // SDPA program config
    auto sdpa_config = ttnn::operations::transformer::SDPAProgramConfig{
        .compute_with_storage_grid_size = grid,
        .sub_core_grids = std::nullopt,
        .q_chunk_size = 64,  // Match our seq length
        .k_chunk_size = 64,
        .exp_approx_mode = false,
        .max_cores_per_head_batch = 16,
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

        float scale = 1.0f / std::sqrt(static_cast<float>(HEAD_DIM));

        // =====================================================================
        // WARMUP
        // =====================================================================
        fmt::print("Warming up ({} iters each)...\n", WARMUP_ITERS);

        for (int i = 0; i < WARMUP_ITERS; ++i) {
            // Manual attention warmup
            auto q_proj = ttnn::matmul(x, wq, false, true, std::nullopt, std::nullopt,
                                       std::nullopt, std::nullopt, compute_config);
            auto k_proj = ttnn::matmul(x, wk, false, true, std::nullopt, std::nullopt,
                                       std::nullopt, std::nullopt, compute_config);
            auto v_proj = ttnn::matmul(x, wv, false, true, std::nullopt, std::nullopt,
                                       std::nullopt, std::nullopt, compute_config);
            auto q = ttnn::transpose(ttnn::reshape(q_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
            auto k = ttnn::transpose(ttnn::reshape(k_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
            auto v = ttnn::transpose(ttnn::reshape(v_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
            auto scores = ttnn::multiply(ttnn::matmul(q, k, false, true, std::nullopt, std::nullopt,
                                                       std::nullopt, std::nullopt, compute_config), scale);
            auto scores_masked = ttnn::add(scores, mask);
            auto attn_weights = ttnn::softmax(scores_masked, -1);
            auto attn_out = ttnn::matmul(attn_weights, v, false, false, std::nullopt, std::nullopt,
                                          std::nullopt, std::nullopt, compute_config);
            auto attn_merged = ttnn::reshape(ttnn::transpose(attn_out, 1, 2), ttnn::Shape({BATCH, SEQ, DIM}));
            auto output = ttnn::matmul(attn_merged, wo, false, true, std::nullopt, std::nullopt,
                                        std::nullopt, std::nullopt, compute_config);
            sync();
        }

        for (int i = 0; i < WARMUP_ITERS; ++i) {
            // SDPA warmup - do projections and reshape first
            auto q_proj = ttnn::matmul(x, wq, false, true, std::nullopt, std::nullopt,
                                       std::nullopt, std::nullopt, compute_config);
            auto k_proj = ttnn::matmul(x, wk, false, true, std::nullopt, std::nullopt,
                                       std::nullopt, std::nullopt, compute_config);
            auto v_proj = ttnn::matmul(x, wv, false, true, std::nullopt, std::nullopt,
                                       std::nullopt, std::nullopt, compute_config);
            auto q = ttnn::transpose(ttnn::reshape(q_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
            auto k = ttnn::transpose(ttnn::reshape(k_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
            auto v = ttnn::transpose(ttnn::reshape(v_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);

            // Fused SDPA
            auto attn_out = ttnn::transformer::scaled_dot_product_attention(
                q, k, v,
                std::nullopt,  // attn_mask (use is_causal instead)
                true,          // is_causal
                scale,         // scale
                std::nullopt,  // sliding_window
                std::nullopt,  // memory_config
                sdpa_config,   // program_config
                compute_config // compute_kernel_config
            );

            auto attn_merged = ttnn::reshape(ttnn::transpose(attn_out, 1, 2), ttnn::Shape({BATCH, SEQ, DIM}));
            auto output = ttnn::matmul(attn_merged, wo, false, true, std::nullopt, std::nullopt,
                                        std::nullopt, std::nullopt, compute_config);
            sync();
        }

        // =====================================================================
        // TIMED RUNS - MANUAL ATTENTION
        // =====================================================================
        fmt::print("\nTiming MANUAL attention ({} iters)...\n", TIMED_ITERS);
        double manual_total_us = 0;
        Tensor manual_output;

        for (int iter = 0; iter < TIMED_ITERS; ++iter) {
            sync();
            auto start = std::chrono::high_resolution_clock::now();

            // QKV projections
            auto q_proj = ttnn::matmul(x, wq, false, true, std::nullopt, std::nullopt,
                                       std::nullopt, std::nullopt, compute_config);
            auto k_proj = ttnn::matmul(x, wk, false, true, std::nullopt, std::nullopt,
                                       std::nullopt, std::nullopt, compute_config);
            auto v_proj = ttnn::matmul(x, wv, false, true, std::nullopt, std::nullopt,
                                       std::nullopt, std::nullopt, compute_config);

            // Reshape for multi-head
            auto q = ttnn::transpose(ttnn::reshape(q_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
            auto k = ttnn::transpose(ttnn::reshape(k_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
            auto v = ttnn::transpose(ttnn::reshape(v_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);

            // Attention scores
            auto scores = ttnn::multiply(ttnn::matmul(q, k, false, true, std::nullopt, std::nullopt,
                                                       std::nullopt, std::nullopt, compute_config), scale);
            auto scores_masked = ttnn::add(scores, mask);
            auto attn_weights = ttnn::softmax(scores_masked, -1);
            auto attn_out = ttnn::matmul(attn_weights, v, false, false, std::nullopt, std::nullopt,
                                          std::nullopt, std::nullopt, compute_config);

            // Merge heads and output projection
            auto attn_merged = ttnn::reshape(ttnn::transpose(attn_out, 1, 2), ttnn::Shape({BATCH, SEQ, DIM}));
            manual_output = ttnn::matmul(attn_merged, wo, false, true, std::nullopt, std::nullopt,
                                          std::nullopt, std::nullopt, compute_config);

            sync();
            auto end = std::chrono::high_resolution_clock::now();
            manual_total_us += std::chrono::duration<double, std::micro>(end - start).count();
        }

        double manual_avg_us = manual_total_us / TIMED_ITERS;
        double manual_tflops = (FULL_ATTENTION_FLOPS / manual_avg_us) / 1e6;

        // =====================================================================
        // TIMED RUNS - SDPA
        // =====================================================================
        fmt::print("Timing SDPA attention ({} iters)...\n", TIMED_ITERS);
        double sdpa_total_us = 0;
        Tensor sdpa_output;

        for (int iter = 0; iter < TIMED_ITERS; ++iter) {
            sync();
            auto start = std::chrono::high_resolution_clock::now();

            // QKV projections (same as manual)
            auto q_proj = ttnn::matmul(x, wq, false, true, std::nullopt, std::nullopt,
                                       std::nullopt, std::nullopt, compute_config);
            auto k_proj = ttnn::matmul(x, wk, false, true, std::nullopt, std::nullopt,
                                       std::nullopt, std::nullopt, compute_config);
            auto v_proj = ttnn::matmul(x, wv, false, true, std::nullopt, std::nullopt,
                                       std::nullopt, std::nullopt, compute_config);

            // Reshape for multi-head (same as manual)
            auto q = ttnn::transpose(ttnn::reshape(q_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
            auto k = ttnn::transpose(ttnn::reshape(k_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);
            auto v = ttnn::transpose(ttnn::reshape(v_proj, ttnn::Shape({BATCH, SEQ, HEADS, HEAD_DIM})), 1, 2);

            // FUSED SDPA - replaces: Q@K.T, scale, mask, softmax, @V
            auto attn_out = ttnn::transformer::scaled_dot_product_attention(
                q, k, v,
                std::nullopt,  // attn_mask (use is_causal instead)
                true,          // is_causal
                scale,         // scale
                std::nullopt,  // sliding_window
                std::nullopt,  // memory_config
                sdpa_config,   // program_config
                compute_config // compute_kernel_config
            );

            // Merge heads and output projection (same as manual)
            auto attn_merged = ttnn::reshape(ttnn::transpose(attn_out, 1, 2), ttnn::Shape({BATCH, SEQ, DIM}));
            sdpa_output = ttnn::matmul(attn_merged, wo, false, true, std::nullopt, std::nullopt,
                                        std::nullopt, std::nullopt, compute_config);

            sync();
            auto end = std::chrono::high_resolution_clock::now();
            sdpa_total_us += std::chrono::duration<double, std::micro>(end - start).count();
        }

        double sdpa_avg_us = sdpa_total_us / TIMED_ITERS;
        double sdpa_tflops = (FULL_ATTENTION_FLOPS / sdpa_avg_us) / 1e6;

        // =====================================================================
        // ACCURACY COMPARISON
        // =====================================================================
        fmt::print("\nComputing error between SDPA and Manual outputs...\n");

        // Save both outputs for external comparison
        traced::save_tensor(manual_output, OUTPUT_DIR + "/manual_output.bin", device.get());
        traced::save_tensor(sdpa_output, OUTPUT_DIR + "/sdpa_output.bin", device.get());

        // Compute error directly
        sync();
        auto manual_vec = manual_output.cpu().to_vector<bfloat16>();
        auto sdpa_vec = sdpa_output.cpu().to_vector<bfloat16>();

        double max_diff = 0.0;
        double sum_diff = 0.0;
        double sum_abs = 0.0;

        for (size_t i = 0; i < manual_vec.size(); ++i) {
            float m = static_cast<float>(manual_vec[i]);
            float s = static_cast<float>(sdpa_vec[i]);
            double diff = std::abs(m - s);
            max_diff = std::max(max_diff, diff);
            sum_diff += diff;
            sum_abs += std::abs(m);
        }

        double mean_diff = sum_diff / manual_vec.size();
        double rel_error = (sum_abs > 0) ? (sum_diff / sum_abs) * 100.0 : 0.0;

        // =====================================================================
        // RESULTS
        // =====================================================================
        fmt::print("\n");
        fmt::print("==========================================================================================\n");
        fmt::print("RESULTS: SDPA vs Manual Attention\n");
        fmt::print("==========================================================================================\n");
        fmt::print("\n");
        fmt::print("TIMING COMPARISON:\n");
        fmt::print("------------------------------------------------------------------------------------------\n");
        fmt::print("{:<20} {:>15} {:>15} {:>15}\n", "Method", "Avg Time (us)", "TFLOPS", "Speedup");
        fmt::print("------------------------------------------------------------------------------------------\n");
        fmt::print("{:<20} {:>15.2f} {:>15.4f} {:>15}\n", "Manual", manual_avg_us, manual_tflops, "1.00x");
        fmt::print("{:<20} {:>15.2f} {:>15.4f} {:>15.2f}x\n", "SDPA (fused)", sdpa_avg_us, sdpa_tflops, manual_avg_us / sdpa_avg_us);
        fmt::print("------------------------------------------------------------------------------------------\n");
        fmt::print("\n");
        fmt::print("ACCURACY (SDPA vs Manual):\n");
        fmt::print("------------------------------------------------------------------------------------------\n");
        fmt::print("  Max difference:  {:.6f}\n", max_diff);
        fmt::print("  Mean difference: {:.6f}\n", mean_diff);
        fmt::print("  Relative error:  {:.4f}%\n", rel_error);
        fmt::print("------------------------------------------------------------------------------------------\n");
        fmt::print("\n");
        fmt::print("FLOPs BREAKDOWN (for reference):\n");
        fmt::print("------------------------------------------------------------------------------------------\n");
        fmt::print("  Attention core (Q@K.T, scale, mask, softmax, @V): {:.2f} MFLOPs\n", ATTENTION_CORE_FLOPS / 1e6);
        fmt::print("  QKV + Output projections:                         {:.2f} MFLOPs\n", (FULL_ATTENTION_FLOPS - ATTENTION_CORE_FLOPS) / 1e6);
        fmt::print("  Full attention total:                             {:.2f} MFLOPs\n", FULL_ATTENTION_FLOPS / 1e6);
        fmt::print("==========================================================================================\n");

        // Also save comparison for Python analysis
        fmt::print("\nSaved outputs to {}:\n", OUTPUT_DIR);
        fmt::print("  - manual_output.bin\n");
        fmt::print("  - sdpa_output.bin\n");

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
