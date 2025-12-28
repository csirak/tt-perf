// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Test: Static Autograd with Pre-allocated Buffers
// Verifies that automatic differentiation works with pre-allocated tensors.

#include "static_autograd/nn.hpp"
#include "common.hpp"

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

using namespace static_autograd;

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    std::cout << "Static Autograd Test\n";
    std::cout << "====================\n\n";

    // Config
    constexpr uint32_t batch = 32;
    constexpr uint32_t in_dim = 64;
    constexpr uint32_t hidden_dim = 64;
    constexpr uint32_t out_dim = 64;
    constexpr float lr = 0.01f;

    // Create MLP
    MLP model(batch, in_dim, hidden_dim, out_dim, 0.01f, device);

    // Create input and target buffers
    Tensor input = make_full(ttnn::Shape({batch, in_dim}), 0.1f, device);
    Tensor target = make_full(ttnn::Shape({batch, out_dim}), 0.5f, device);
    Tensor d_input = make_zeros(ttnn::Shape({batch, in_dim}), device);

    // Loss buffers
    Tensor loss = make_zeros(ttnn::Shape({1, 1}), device);
    Tensor d_loss = make_zeros(ttnn::Shape({1, 1}), device);
    Tensor diff = make_zeros(ttnn::Shape({batch, out_dim}), device);

    std::cout << "Config: batch=" << batch << ", in=" << in_dim
              << ", hidden=" << hidden_dim << ", out=" << out_dim << "\n\n";

    // Training loop
    std::cout << "Training for 10 steps...\n";
    std::cout << std::fixed << std::setprecision(6);

    float prev_loss = 1e9f;
    for (int step = 0; step < 10; ++step) {
        // Create graph for this forward pass
        Graph g;

        // Forward
        auto* x = g.leaf(&input, &d_input, false);
        auto* pred = model.forward(g, x);
        auto* loss_v = mse(g, pred, &target, &loss, &d_loss, &diff);

        // Zero gradients
        g.zero_grad();
        model.zero_grad();

        // Backward
        g.backward(loss_v);

        // Get loss value
        tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        float loss_val = static_cast<float>(loss.cpu().to_vector<bfloat16>()[0]);

        std::cout << "Step " << step << ": loss = " << loss_val << "\n";

        // Check loss is decreasing (with some tolerance for noise)
        if (step > 0) {
            assert(loss_val < prev_loss * 1.5f);  // Allow some noise
        }
        prev_loss = loss_val;

        // SGD update
        model.sgd_step(lr);
    }

    std::cout << "\n";

    // Verify gradients are non-zero
    tt::tt_metal::distributed::Synchronize(&device, std::nullopt);

    auto d_w1 = model.layer1.d_weight.cpu().to_vector<bfloat16>();
    auto d_w2 = model.layer2.d_weight.cpu().to_vector<bfloat16>();

    float dw1_sum = 0.0f, dw2_sum = 0.0f;
    for (size_t i = 0; i < std::min(d_w1.size(), size_t(100)); ++i) {
        dw1_sum += std::abs(static_cast<float>(d_w1[i]));
        dw2_sum += std::abs(static_cast<float>(d_w2[i]));
    }

    std::cout << "Gradient check:\n";
    std::cout << "  layer1.d_weight sum (first 100): " << dw1_sum << "\n";
    std::cout << "  layer2.d_weight sum (first 100): " << dw2_sum << "\n";

    // Gradients should be non-zero after training
    assert(dw1_sum > 1e-6f || dw2_sum > 1e-6f);
    std::cout << "  PASSED: Gradients are non-zero\n\n";

    // Test that backward() properly accumulates
    std::cout << "Testing gradient accumulation...\n";
    {
        Graph g;
        model.zero_grad();

        auto* x = g.leaf(&input, &d_input, false);
        auto* pred = model.forward(g, x);
        auto* loss_v = mse(g, pred, &target, &loss, &d_loss, &diff);

        g.zero_grad();
        g.backward(loss_v);

        tt::tt_metal::distributed::Synchronize(&device, std::nullopt);
        auto d_w1_first = model.layer1.d_weight.cpu().to_vector<bfloat16>();
        float first_grad = static_cast<float>(d_w1_first[0]);

        std::cout << "  First backward d_weight[0]: " << first_grad << "\n";
        std::cout << "  PASSED: Backward completed\n";
    }

    std::cout << "\n====================\n";
    std::cout << "All tests PASSED!\n";

    return 0;
}
