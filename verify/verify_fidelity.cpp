// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Math Fidelity Comparison Test
// Compare HiFi4, HiFi2, and LoFi against PyTorch reference values.

#include "common.hpp"
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using Tensor = tt::tt_metal::Tensor;
// MathFidelity is in global namespace (from base_types.hpp)

// Helper to create constant tensor
inline Tensor make_full(ttnn::Shape shape, float value, MeshDevice& device) {
    return ttnn::full(shape, value, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
}

// MLP with configurable math fidelity
struct FidelityMLP {
    Tensor w1, w2, b1, b2;
    Tensor h1, h1_relu, pred;
    Tensor diff, sq, loss;
    Tensor d_sq, d_diff, d_pred;
    Tensor d_h1_relu, d_h1, relu_mask;
    Tensor d_w1, d_w2, d_b1, d_b2;

    uint32_t batch, in_features, hidden, out_features;
    float lr, loss_scale;
    std::optional<ttnn::DeviceComputeKernelConfig> compute_config;

    FidelityMLP(uint32_t batch_, uint32_t in_, uint32_t hid_, uint32_t out_,
                float lr_, MathFidelity fid_, MeshDevice& device)
        : batch(batch_), in_features(in_), hidden(hid_), out_features(out_), lr(lr_) {

        loss_scale = 1.0f / (batch * out_features);

        // Set up compute config with specified fidelity
        ttnn::WormholeComputeKernelConfig wh_config;
        wh_config.math_fidelity = fid_;
        wh_config.math_approx_mode = true;
        wh_config.fp32_dest_acc_en = false;
        wh_config.packer_l1_acc = false;
        compute_config = wh_config;

        // Initialize weights - use small constant that won't overflow with large matmuls
        // For dims 2048->1024->512: values will be ~2048*0.001 = 2.0 after first layer
        w1 = make_full(ttnn::Shape({in_features, hidden}), 0.001f, device);
        w2 = make_full(ttnn::Shape({hidden, out_features}), 0.001f, device);
        b1 = make_full(ttnn::Shape({1, hidden}), 0.0f, device);
        b2 = make_full(ttnn::Shape({1, out_features}), 0.0f, device);
    }

    void forward(const Tensor& x) {
        h1 = ttnn::add(ttnn::matmul(x, w1, false, false, std::nullopt, std::nullopt,
                                     std::nullopt, std::nullopt, compute_config), b1);
        relu_mask = ttnn::gtz(h1);
        h1_relu = ttnn::relu(h1);
        pred = ttnn::add(ttnn::matmul(h1_relu, w2, false, false, std::nullopt, std::nullopt,
                                       std::nullopt, std::nullopt, compute_config), b2);
    }

    void compute_loss(const Tensor& target) {
        diff = ttnn::subtract(pred, target);
        sq = ttnn::multiply(diff, diff);
        loss = ttnn::mean(sq, std::nullopt, true);
    }

    void backward(const Tensor& x) {
        d_sq = ttnn::full_like(sq, loss_scale);
        d_diff = ttnn::multiply(d_sq, ttnn::multiply(diff, 2.0f));
        d_pred = d_diff;

        d_w2 = ttnn::matmul(h1_relu, d_pred, true, false, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, compute_config);
        d_b2 = ttnn::sum(d_pred, 0, true);
        d_h1_relu = ttnn::matmul(d_pred, w2, false, true, std::nullopt, std::nullopt,
                                  std::nullopt, std::nullopt, compute_config);

        d_h1 = ttnn::multiply(d_h1_relu, relu_mask);

        d_w1 = ttnn::matmul(x, d_h1, true, false, std::nullopt, std::nullopt,
                            std::nullopt, std::nullopt, compute_config);
        d_b1 = ttnn::sum(d_h1, 0, true);
    }

    void sgd_step() {
        w1 = ttnn::subtract(w1, ttnn::multiply(d_w1, lr));
        w2 = ttnn::subtract(w2, ttnn::multiply(d_w2, lr));
        b1 = ttnn::subtract(b1, ttnn::multiply(d_b1, lr));
        b2 = ttnn::subtract(b2, ttnn::multiply(d_b2, lr));
    }

    void train_step(const Tensor& x, const Tensor& target) {
        forward(x);
        compute_loss(target);
        backward(x);
        sgd_step();
    }

    float get_loss(MeshDevice* dev) {
        tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        return static_cast<float>(loss.cpu().to_vector<bfloat16>()[0]);
    }

    float get_w1_00(MeshDevice* dev) {
        tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        return static_cast<float>(w1.cpu().to_vector<bfloat16>()[0]);
    }
};

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();
    auto* dev = &device;

    // Config - very large weight matrices to see fidelity performance difference
    const uint32_t batch = 512;
    const uint32_t in_features = 8192;
    const uint32_t hidden = 4096;
    const uint32_t out_features = 2048;
    const float lr = 0.1f;
    const int num_steps = 10;

    // Note: PyTorch reference values not available for this config
    // Focus is on performance comparison between fidelity levels

    std::cout << "Math Fidelity Comparison Test\n";
    std::cout << "==============================\n";
    std::cout << "Config: batch=" << batch << ", " << in_features << "->" << hidden << "->" << out_features << "\n";
    std::cout << "Learning rate: " << lr << "\n\n";

    // Inputs - scale x to avoid overflow with large dimensions
    // Use 1/sqrt(in_features) so first layer output is O(1)
    float x_scale = 1.0f / std::sqrt(static_cast<float>(in_features));
    auto x = make_full(ttnn::Shape({batch, in_features}), x_scale, device);
    auto target = make_full(ttnn::Shape({batch, out_features}), 0.5f, device);

    // Test each fidelity
    std::vector<std::pair<MathFidelity, std::string>> fidelities = {
        {MathFidelity::HiFi4, "HiFi4"},
        {MathFidelity::HiFi2, "HiFi2"},
        {MathFidelity::LoFi, "LoFi"}
    };

    // Store results and timing
    std::vector<std::vector<float>> results(fidelities.size());
    std::vector<double> avg_step_times(fidelities.size());
    const int warmup_steps = 2;
    const int timed_steps = 20;

    for (size_t f = 0; f < fidelities.size(); ++f) {
        auto [fidelity, name] = fidelities[f];
        std::cout << "Testing " << name << "...\n";

        FidelityMLP model(batch, in_features, hidden, out_features, lr, fidelity, device);

        // Initial loss
        model.forward(x);
        model.compute_loss(target);
        results[f].push_back(model.get_loss(dev));

        // Training steps for loss comparison
        for (int step = 1; step <= num_steps; ++step) {
            model.forward(x);
            model.compute_loss(target);
            model.backward(x);
            model.sgd_step();

            model.forward(x);
            model.compute_loss(target);
            results[f].push_back(model.get_loss(dev));
        }

        // Warmup for timing
        for (int i = 0; i < warmup_steps; ++i) {
            model.forward(x);
            model.compute_loss(target);
            model.backward(x);
            model.sgd_step();
        }
        tt::tt_metal::distributed::Synchronize(dev, std::nullopt);

        // Timed steps
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < timed_steps; ++i) {
            model.forward(x);
            model.compute_loss(target);
            model.backward(x);
            model.sgd_step();
        }
        tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        auto end = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
        avg_step_times[f] = total_ms / timed_steps;
    }

    // Print comparison table (comparing fidelities against HiFi4 as baseline)
    std::cout << "\n" << std::string(75, '=') << "\n";
    std::cout << "Loss Comparison (HiFi4 as baseline)\n";
    std::cout << std::string(75, '=') << "\n\n";

    std::cout << std::fixed << std::setprecision(6);
    std::cout << std::setw(6) << "Step"
              << std::setw(14) << "HiFi4"
              << std::setw(14) << "HiFi2"
              << std::setw(14) << "LoFi"
              << std::setw(12) << "HiFi2 diff"
              << std::setw(12) << "LoFi diff"
              << "\n";
    std::cout << std::string(75, '-') << "\n";

    for (int step = 0; step <= num_steps; ++step) {
        float hifi4 = results[0][step];
        float hifi2 = results[1][step];
        float lofi = results[2][step];

        // Calculate percentage difference from HiFi4
        auto pct_diff = [](float ref, float val) {
            if (ref == 0) return 0.0f;
            return std::abs(val - ref) / ref * 100.0f;
        };

        std::cout << std::setw(6) << step
                  << std::setw(14) << hifi4
                  << std::setw(14) << hifi2
                  << std::setw(14) << lofi
                  << std::setw(10) << std::setprecision(1) << pct_diff(hifi4, hifi2) << "%"
                  << std::setw(10) << pct_diff(hifi4, lofi) << "%"
                  << std::setprecision(6)
                  << "\n";
    }

    std::cout << "\n" << std::string(75, '=') << "\n";
    std::cout << "Legend:\n";
    std::cout << "  HiFi4: Highest precision (default for matmul)\n";
    std::cout << "  HiFi2: Medium precision\n";
    std::cout << "  LoFi:  Lowest precision - fastest but may diverge\n";

    // Print timing comparison
    std::cout << "\n" << std::string(90, '=') << "\n";
    std::cout << "Performance Comparison (avg of " << timed_steps << " steps after " << warmup_steps << " warmup)\n";
    std::cout << std::string(90, '=') << "\n\n";

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  HiFi4: " << avg_step_times[0] << " ms/step\n";
    std::cout << "  HiFi2: " << avg_step_times[1] << " ms/step";
    if (avg_step_times[1] < avg_step_times[0]) {
        std::cout << "  (" << std::setprecision(2) << (avg_step_times[0] / avg_step_times[1]) << "x faster)";
    }
    std::cout << "\n";
    std::cout << "  LoFi:  " << std::setprecision(3) << avg_step_times[2] << " ms/step";
    if (avg_step_times[2] < avg_step_times[0]) {
        std::cout << "  (" << std::setprecision(2) << (avg_step_times[0] / avg_step_times[2]) << "x faster)";
    }
    std::cout << "\n";

    return 0;
}
