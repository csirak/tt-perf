// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// TTML Scaling Sweep Benchmark: Batch x Dimension
// Measures non-traced execution (TTML tracing has issues).
// Outputs CSV for comparison with our traced autograd.

#include <fmt/format.h>
#include <vector>
#include <chrono>
#include <fstream>

#include <core/ttnn_all_includes.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <tt-metalium/distributed.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"
#include "ops/linear_op.hpp"
#include "ops/unary_ops.hpp"

using ttml::autograd::TensorPtr;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;

// Simple 2-layer MLP with constant init
struct SimpleMLP {
    TensorPtr w1, b1, w2, b2;

    SimpleMLP(uint32_t dim, float init_val, MeshDevice* device) {
        // Layer 1: dim -> dim
        auto w1_tensor = ttnn::full(ttnn::Shape({1, 1, dim, dim}), init_val,
                                    ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device);
        w1 = ttml::autograd::create_tensor(w1_tensor);

        auto b1_tensor = ttnn::zeros(ttnn::Shape({1, 1, 1, dim}),
                                     ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device);
        b1 = ttml::autograd::create_tensor(b1_tensor);

        // Layer 2: dim -> dim
        auto w2_tensor = ttnn::full(ttnn::Shape({1, 1, dim, dim}), init_val,
                                    ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device);
        w2 = ttml::autograd::create_tensor(w2_tensor);

        auto b2_tensor = ttnn::zeros(ttnn::Shape({1, 1, 1, dim}),
                                     ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, *device);
        b2 = ttml::autograd::create_tensor(b2_tensor);
    }

    TensorPtr operator()(const TensorPtr& x) {
        auto h = ttml::ops::linear_op(x, w1, b1);
        h = ttml::ops::relu(h);
        return ttml::ops::linear_op(h, w2, b2);
    }

    std::vector<TensorPtr> parameters() {
        return {w1, b1, w2, b2};
    }
};

// Simple SGD step (no momentum)
void sgd_step(std::vector<TensorPtr>& params, float lr) {
    for (auto& p : params) {
        if (p->is_grad_initialized()) {
            auto val = p->get_value();
            auto grad = p->get_grad();
            p->set_value(ttnn::subtract(val, ttnn::multiply(grad, lr)));
        }
    }
}

// Run benchmark for given batch and dim
double benchmark_config(uint32_t batch, uint32_t dim, MeshDevice* device, int warmup, int iters) {
    const float lr = 0.01f;
    const float init_val = 0.01f;

    // Create model
    SimpleMLP model(dim, init_val, device);
    auto params = model.parameters();

    // Create input and target (4D shapes for TTML)
    std::vector<float> input_data(batch * dim, 1.0f);
    std::vector<float> target_data(batch * dim, 0.5f);

    auto input = ttml::autograd::create_tensor(
        ttml::core::from_vector(input_data, ttnn::Shape({batch, 1, 1, dim}), device));
    auto target = ttml::autograd::create_tensor(
        ttml::core::from_vector(target_data, ttnn::Shape({batch, 1, 1, dim}), device));

    // Train step lambda
    auto train_step = [&]() {
        auto output = model(input);
        auto loss = ttml::ops::mse_loss(output, target);
        loss->backward();
        sgd_step(params, lr);
        ttml::autograd::ctx().reset_graph();
        return loss;
    };

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        train_step();
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);

    // Measure non-traced execution
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        train_step();
    }
    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / iters;
}

int main() {
    fmt::print("TTML Scaling Sweep Benchmark (Non-traced)\n");
    fmt::print("==========================================\n");

    constexpr int warmup = 5;
    constexpr int iters = 20;

    // Sweep parameters (same as our autograd)
    const uint32_t batches[] = {256, 512, 1024, 2048, 4096};
    const uint32_t dims[] = {256, 512, 1024, 2048};

    fmt::print("Warmup: {}, Iters: {}\n\n", warmup, iters);

    auto* device = &ttml::autograd::ctx().get_device();
    fmt::print("Device initialized\n\n");

    // Open CSV file
    std::ofstream csv("bench_sweep_ttml.csv");
    csv << "batch,dim,ms_per_iter\n";

    for (auto batch : batches) {
        for (auto dim : dims) {
            fmt::print("batch={}, dim={} ... ", batch, dim);

            try {
                double ms = benchmark_config(batch, dim, device, warmup, iters);
                csv << batch << "," << dim << "," << fmt::format("{:.3f}", ms) << "\n";
                csv.flush();
                fmt::print("{:.3f} ms/iter\n", ms);
            } catch (const std::exception& e) {
                csv << batch << "," << dim << ",OOM\n";
                csv.flush();
                fmt::print("OOM or error: {}\n", e.what());
                try {
                    tt::tt_metal::distributed::Synchronize(device, std::nullopt);
                } catch (...) {}
            }
        }
    }

    csv.close();
    fmt::print("\nResults written to bench_sweep_ttml.csv\n");

    // Must close device to ensure clean shutdown
    ttml::autograd::ctx().close_device();

    return 0;
}
