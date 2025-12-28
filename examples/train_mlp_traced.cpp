// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Traced MLP Training - Verification Version
// Prints step-by-step loss to compare with PyTorch reference.
// Config: batch=1024, 512->256->128, init=0.01, lr=0.01

#include "traced/ops.hpp"
#include "traced/nn.hpp"
#include "traced/trace.hpp"
#include "common.hpp"

#include <iostream>
#include <iomanip>

using MeshDevice = tt::tt_metal::distributed::MeshDevice;
using Tensor = tt::tt_metal::Tensor;

// Helper to get mean of tensor
float get_mean(const Tensor& t, MeshDevice* dev) {
    tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
    auto vec = t.cpu().to_vector<bfloat16>();
    float sum = 0;
    for (auto v : vec) sum += static_cast<float>(v);
    return sum / vec.size();
}

// Helper to get first element of tensor
float get_first(const Tensor& t, MeshDevice* dev) {
    tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
    return static_cast<float>(t.cpu().to_vector<bfloat16>()[0]);
}

// TracedMLP: 2-layer MLP with all buffers pre-allocated
struct TracedMLP {
    traced::TracedLinear fc1;
    traced::TracedLinear fc2;
    Tensor h1;
    Tensor h1_relu;
    Tensor relu_mask;
    Tensor pred;
    Tensor diff;
    Tensor sq;
    Tensor loss;
    Tensor d_pred;
    Tensor d_h1_relu;
    Tensor d_h1;
    float loss_scale;
    float lr;

    TracedMLP(uint32_t batch, uint32_t in_f, uint32_t hid, uint32_t out_f,
              float learning_rate, MeshDevice& device)
        : fc1(in_f, hid, 0.01f, device),
          fc2(hid, out_f, 0.01f, device),
          h1(traced::make_zeros(ttnn::Shape({batch, hid}), device)),
          h1_relu(traced::make_zeros(ttnn::Shape({batch, hid}), device)),
          relu_mask(traced::make_zeros(ttnn::Shape({batch, hid}), device)),
          pred(traced::make_zeros(ttnn::Shape({batch, out_f}), device)),
          diff(traced::make_zeros(ttnn::Shape({batch, out_f}), device)),
          sq(traced::make_zeros(ttnn::Shape({batch, out_f}), device)),
          loss(traced::make_zeros(ttnn::Shape({1, 1}), device)),
          d_pred(traced::make_zeros(ttnn::Shape({batch, out_f}), device)),
          d_h1_relu(traced::make_zeros(ttnn::Shape({batch, hid}), device)),
          d_h1(traced::make_zeros(ttnn::Shape({batch, hid}), device)),
          loss_scale(1.0f / (batch * out_f)),
          lr(learning_rate) {}

    void forward(const Tensor& x) {
        fc1.forward(x, h1);
        traced::relu(h1, h1_relu, relu_mask);
        fc2.forward(h1_relu, pred);
    }

    void compute_loss(const Tensor& target) {
        traced::subtract(pred, target, diff);
        traced::multiply(diff, diff, sq);
        traced::mean(sq, loss);
    }

    void backward(const Tensor& x) {
        traced::mse_backward(diff, loss_scale, d_pred);
        fc2.backward(h1_relu, d_pred, d_h1_relu);
        traced::relu_backward(d_h1_relu, relu_mask, d_h1);
        fc1.backward_no_input(x, d_h1);
    }

    void sgd_step() {
        fc1.sgd_step(lr);
        fc2.sgd_step(lr);
    }

    float get_loss(MeshDevice* dev) {
        tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        return static_cast<float>(loss.cpu().to_vector<bfloat16>()[0]);
    }
};

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();
    auto* dev = &device;

    // Configuration
    const uint32_t batch = 1024;
    const uint32_t in_features = 512;
    const uint32_t hidden = 256;
    const uint32_t out_features = 128;
    const float lr = 0.01f;
    const int num_steps = 5;

    std::cout << "=== C++ Traced Autograd MLP Verification ===\n\n";
    std::cout << "Config:\n";
    std::cout << "  batch_size: " << batch << "\n";
    std::cout << "  in_features: " << in_features << "\n";
    std::cout << "  hidden_features: " << hidden << "\n";
    std::cout << "  out_features: " << out_features << "\n";
    std::cout << "  learning_rate: " << lr << "\n";
    std::cout << "  init_val: 0.01\n";
    std::cout << "  num_steps: " << num_steps << "\n\n";

    // Create input and target tensors
    auto x = traced::make_full(ttnn::Shape({batch, in_features}), 1.0f, device);
    auto target = traced::make_full(ttnn::Shape({batch, out_features}), 0.5f, device);

    // Create model
    TracedMLP model(batch, in_features, hidden, out_features, lr, device);

    // Initial forward (before training)
    std::cout << "Initial state (before training):\n";
    model.forward(x);
    model.compute_loss(target);

    std::cout << "  Output mean: " << std::fixed << std::setprecision(6)
              << get_mean(model.pred, dev) << "\n";
    std::cout << "  Output[0,0]: " << get_first(model.pred, dev) << "\n";
    std::cout << "  Initial loss: " << model.get_loss(dev) << "\n";
    std::cout << "  W1[0,0]: " << get_first(model.fc1.weight, dev) << "\n";
    std::cout << "  W2[0,0]: " << get_first(model.fc2.weight, dev) << "\n";

    // Also check first backward to see gradients
    model.backward(x);
    std::cout << "\nGradients at step 0:\n";
    std::cout << "  dW1[0,0]: " << get_first(model.fc1.d_weight, dev) << "\n";
    std::cout << "  dW1 mean: " << get_mean(model.fc1.d_weight, dev) << "\n";
    std::cout << "  dW2[0,0]: " << get_first(model.fc2.d_weight, dev) << "\n";
    std::cout << "  dW2 mean: " << get_mean(model.fc2.d_weight, dev) << "\n";
    std::cout << "  db1[0]: " << get_first(model.fc1.d_bias, dev) << "\n";
    std::cout << "  db2[0]: " << get_first(model.fc2.d_bias, dev) << "\n";

    // Re-create model for clean training
    TracedMLP model2(batch, in_features, hidden, out_features, lr, device);

    std::cout << "\nTraining steps:\n";
    std::cout << std::string(60, '-') << "\n";

    for (int step = 0; step < num_steps; ++step) {
        // Forward
        model2.forward(x);
        model2.compute_loss(target);
        float loss_before = model2.get_loss(dev);

        // Backward
        model2.backward(x);

        // SGD step
        model2.sgd_step();

        // Get loss after update
        model2.forward(x);
        model2.compute_loss(target);
        float loss_after = model2.get_loss(dev);

        std::cout << "Step " << step << ": loss_before=" << std::fixed << std::setprecision(6)
                  << loss_before << ", loss_after=" << loss_after << "\n";
    }

    std::cout << "\nFinal state:\n";
    std::cout << "  W1[0,0]: " << get_first(model2.fc1.weight, dev) << "\n";
    std::cout << "  W2[0,0]: " << get_first(model2.fc2.weight, dev) << "\n";
    std::cout << "  Final loss: " << model2.get_loss(dev) << "\n";

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "EXPECTED VALUES (from PyTorch bf16):\n";
    std::cout << "  Initial loss: 159.390625\n";
    std::cout << "  Final loss: 0.000244\n";
    std::cout << std::string(60, '=') << "\n";

    return 0;
}
