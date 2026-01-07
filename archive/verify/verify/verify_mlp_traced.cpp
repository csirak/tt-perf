// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MLP Verification Script (TTNN C++)
// Uses deterministic inputs to verify correctness against PyTorch implementation.
//
// All values are chosen to match exactly with verify_mlp.py.

#include "common.hpp"
#include <ttnn/operations/creation.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <iostream>
#include <iomanip>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using Tensor = tt::tt_metal::Tensor;

// Helper to create constant tensor on device
inline Tensor make_full(ttnn::Shape shape, float value, MeshDevice& device) {
    return ttnn::full(shape, value, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
}

// Verification MLP with deterministic weights
struct VerifyMLP {
    // Weights
    Tensor w1, w2;  // [in, hidden], [hidden, out]
    Tensor b1, b2;  // [1, hidden], [1, out]

    // Forward activations
    Tensor h1, h1_relu, pred;

    // Loss computation
    Tensor diff, sq, loss;

    // Backward gradients
    Tensor d_sq, d_diff, d_pred;
    Tensor d_h1_relu, d_h1, relu_mask;
    Tensor d_w1, d_w2, d_b1, d_b2;

    // Config
    uint32_t batch, in_features, hidden, out_features;
    float lr;
    float loss_scale;

    VerifyMLP(uint32_t batch_, uint32_t in_, uint32_t hid_, uint32_t out_, float lr_, MeshDevice& device)
        : batch(batch_), in_features(in_), hidden(hid_), out_features(out_), lr(lr_) {

        loss_scale = 1.0f / (batch * out_features);

        // Initialize weights to constant 0.01 (matching PyTorch)
        // Note: TTNN uses [in, out] layout, PyTorch uses [out, in]
        // So W1 in PyTorch is [hidden, in_features], in TTNN it's [in_features, hidden]
        w1 = make_full(ttnn::Shape({in_features, hidden}), 0.01f, device);
        w2 = make_full(ttnn::Shape({hidden, out_features}), 0.01f, device);
        b1 = make_full(ttnn::Shape({1, hidden}), 0.0f, device);
        b2 = make_full(ttnn::Shape({1, out_features}), 0.0f, device);
    }

    void forward(const Tensor& x) {
        h1 = ttnn::add(ttnn::matmul(x, w1), b1);
        relu_mask = ttnn::gtz(h1);
        h1_relu = ttnn::relu(h1);
        pred = ttnn::add(ttnn::matmul(h1_relu, w2), b2);
    }

    void compute_loss(const Tensor& target) {
        diff = ttnn::subtract(pred, target);
        sq = ttnn::multiply(diff, diff);
        loss = ttnn::mean(sq, std::nullopt, true);
    }

    void backward(const Tensor& x) {
        // d_sq = 1/numel (gradient of mean)
        d_sq = ttnn::full_like(sq, loss_scale);

        // d_diff = d_sq * 2 * diff
        d_diff = ttnn::multiply(d_sq, ttnn::multiply(diff, 2.0f));
        d_pred = d_diff;

        // Backward through second linear
        d_w2 = ttnn::matmul(h1_relu, d_pred, true, false);
        d_b2 = ttnn::sum(d_pred, 0, true);
        d_h1_relu = ttnn::matmul(d_pred, w2, false, true);

        // Backward through ReLU
        d_h1 = ttnn::multiply(d_h1_relu, relu_mask);

        // Backward through first linear
        d_w1 = ttnn::matmul(x, d_h1, true, false);
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

    float get_loss_value(MeshDevice* device_ptr) {
        tt::tt_metal::distributed::Synchronize(device_ptr, std::nullopt);
        return static_cast<float>(loss.cpu().to_vector<bfloat16>()[0]);
    }

    float get_w1_00(MeshDevice* device_ptr) {
        tt::tt_metal::distributed::Synchronize(device_ptr, std::nullopt);
        return static_cast<float>(w1.cpu().to_vector<bfloat16>()[0]);
    }

    float get_w2_00(MeshDevice* device_ptr) {
        tt::tt_metal::distributed::Synchronize(device_ptr, std::nullopt);
        return static_cast<float>(w2.cpu().to_vector<bfloat16>()[0]);
    }

    float get_dw1_00(MeshDevice* device_ptr) {
        tt::tt_metal::distributed::Synchronize(device_ptr, std::nullopt);
        return static_cast<float>(d_w1.cpu().to_vector<bfloat16>()[0]);
    }

    float get_dw2_00(MeshDevice* device_ptr) {
        tt::tt_metal::distributed::Synchronize(device_ptr, std::nullopt);
        return static_cast<float>(d_w2.cpu().to_vector<bfloat16>()[0]);
    }

    float get_db1_0(MeshDevice* device_ptr) {
        tt::tt_metal::distributed::Synchronize(device_ptr, std::nullopt);
        return static_cast<float>(d_b1.cpu().to_vector<bfloat16>()[0]);
    }

    float get_db2_0(MeshDevice* device_ptr) {
        tt::tt_metal::distributed::Synchronize(device_ptr, std::nullopt);
        return static_cast<float>(d_b2.cpu().to_vector<bfloat16>()[0]);
    }
};

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();
    auto* device_ptr = &device;

    // Configuration - must match Python exactly
    const uint32_t batch_size = 32;
    const uint32_t in_features = 64;
    const uint32_t hidden = 32;
    const uint32_t out_features = 32;
    const float lr = 0.1f;
    const int num_steps = 10;

    std::cout << "MLP Verification (TTNN C++)\n";
    std::cout << std::string(50, '=') << "\n";
    std::cout << "Config: batch=" << batch_size << ", in=" << in_features
              << ", hidden=" << hidden << ", out=" << out_features << "\n";
    std::cout << "Learning rate: " << lr << "\n\n";

    // Deterministic inputs - all ones
    auto x = ttnn::ones(ttnn::Shape({batch_size, in_features}),
                        ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, device);
    auto target = make_full(ttnn::Shape({batch_size, out_features}), 0.5f, device);

    // Create model
    VerifyMLP model(batch_size, in_features, hidden, out_features, lr, device);

    // Compute initial loss (before any training)
    model.forward(x);
    model.compute_loss(target);
    float initial_loss = model.get_loss_value(device_ptr);
    float w1_00 = model.get_w1_00(device_ptr);
    float w2_00 = model.get_w2_00(device_ptr);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Initial loss (step 0): " << initial_loss << "\n";
    std::cout << "  W1[0,0] = " << w1_00 << "\n";
    std::cout << "  W2[0,0] = " << w2_00 << "\n\n";

    // Training loop
    std::cout << "Training steps:\n";
    std::cout << std::string(50, '-') << "\n";

    float final_loss = initial_loss;
    for (int step = 1; step <= num_steps; ++step) {
        // Forward
        model.forward(x);
        model.compute_loss(target);

        // Backward
        model.backward(x);

        // Print gradients at step 1
        if (step == 1) {
            std::cout << "  Gradients at step 1:\n";
            std::cout << "    dW1[0,0] = " << model.get_dw1_00(device_ptr) << "\n";
            std::cout << "    dW2[0,0] = " << model.get_dw2_00(device_ptr) << "\n";
            std::cout << "    db1[0] = " << model.get_db1_0(device_ptr) << "\n";
            std::cout << "    db2[0] = " << model.get_db2_0(device_ptr) << "\n";
        }

        // SGD step
        model.sgd_step();

        // Compute loss after update
        model.forward(x);
        model.compute_loss(target);
        final_loss = model.get_loss_value(device_ptr);
        w1_00 = model.get_w1_00(device_ptr);
        w2_00 = model.get_w2_00(device_ptr);

        std::cout << "Step " << step << ": loss = " << final_loss << "\n";
        std::cout << "  W1[0,0] = " << w1_00 << "\n";
        std::cout << "  W2[0,0] = " << w2_00 << "\n";
    }

    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "Summary:\n";
    std::cout << "  Initial loss: " << initial_loss << "\n";
    std::cout << "  Final loss:   " << final_loss << "\n";
    std::cout << "  Loss reduced: " << (final_loss < initial_loss ? "YES" : "NO") << "\n";

    return 0;
}
