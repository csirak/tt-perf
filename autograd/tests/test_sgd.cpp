// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Test: SGD optimizer with Linear layer

#include "dynamic/autograd.hpp"
#include "dynamic/ops.hpp"
#include "dynamic/nn.hpp"
#include "dynamic/optim.hpp"
#include "common.hpp"

using namespace autograd;

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    // Create Linear layer: in=64, out=32
    Linear linear(64, 32, device);

    // Get initial weight values for comparison
    auto w_before = linear.weight->data.cpu().to_vector<bfloat16>();
    float w0_before = static_cast<float>(w_before[0]);

    // Create SGD optimizer
    SGD sgd(linear.parameters(), 0.01f);

    // Zero gradients (resets to nullopt, no tensor creation)
    sgd.zero_grad();
    assert(!linear.weight->grad.has_value());
    assert(!linear.bias->grad.has_value());

    // Forward pass: x [32, 64] -> y [32, 32]
    auto x = make_ones(ttnn::Shape({32, 64}), device, true, "x");
    auto y = linear.forward(x);
    test::assert_shape(y->data, 32, 32);

    // Backward pass
    backward(y, device);

    // Verify gradients exist
    assert(linear.weight->grad.has_value());
    assert(linear.bias->grad.has_value());

    // dBias = sum(dY, dim=0) = 32 (batch size, since dY = ones)
    test::assert_all_close(linear.bias->grad.value(), 32.0f, 0.5f);

    // SGD step: param = param - lr * grad
    sgd.step();

    // Verify weight changed: new = old - 0.01 * grad
    auto w_after = linear.weight->data.cpu().to_vector<bfloat16>();
    float w0_after = static_cast<float>(w_after[0]);
    assert(std::abs(w0_after - w0_before) > 0.001f);  // Weight changed

    // Verify bias changed: 0 - 0.01 * 32 = -0.32
    test::assert_all_close(linear.bias->data, -0.32f, 0.1f);

    // Second iteration: zero grad, forward, backward, step
    sgd.zero_grad();
    auto y2 = linear.forward(x);
    backward(y2, device);

    // Verify gradients after second backward
    test::assert_all_close(linear.bias->grad.value(), 32.0f, 0.5f);

    sgd.step();

    // Bias should decrease further: -0.32 - 0.01 * 32 = -0.64
    test::assert_all_close(linear.bias->data, -0.64f, 0.1f);

    return 0;
}
