// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Test: Linear layer forward + backward

#include "dynamic/autograd.hpp"
#include "dynamic/ops.hpp"
#include "dynamic/nn.hpp"
#include "common.hpp"

using namespace autograd;

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    // Linear(64, 32): weight [32, 64], bias [1, 32]
    Linear linear(64, 32, device);
    test::assert_shape(linear.weight->data, 32, 64);
    test::assert_shape(linear.bias->data, 1, 32);

    // Verify weight is random (not all same value)
    test::assert_has_variance(linear.weight->data);

    // Forward: x [32, 64] -> y [32, 32]
    auto x = make_random(ttnn::Shape({32, 64}), device, -1.0f, 1.0f, true, "x");
    auto y = linear.forward(x);
    test::assert_shape(y->data, 32, 32);

    // Backward
    backward(y, device);
    assert(x->grad.has_value());
    assert(linear.weight->grad.has_value());
    assert(linear.bias->grad.has_value());

    // dX shape matches input
    test::assert_shape(x->grad.value(), 32, 64);

    // dW shape matches weight
    test::assert_shape(linear.weight->grad.value(), 32, 64);

    // dBias = sum(dY, dim=0) = 32 per element (batch size, since dY = ones)
    test::assert_all_close(linear.bias->grad.value(), 32.0f, 0.1f);

    return 0;
}
