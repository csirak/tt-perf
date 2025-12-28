// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Test: ReLU forward + backward

#include "dynamic/autograd.hpp"
#include "dynamic/ops.hpp"
#include "common.hpp"

using namespace autograd;

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    // Create tensor with all ones (positive values)
    auto x = make_ones(ttnn::Shape({32, 32}), device, true, "x");

    // Forward: relu(ones) = ones (all positive)
    auto y = autograd::relu(x);
    test::assert_shape(y->data, 32, 32);
    assert(y->requires_grad);

    // Verify forward: relu(1.0) = 1.0
    test::assert_all_close(y->data, 1.0f, 0.1f);

    // Backward
    backward(y, device);
    assert(x->grad.has_value());

    // Gradient shape matches input
    test::assert_shape(x->grad.value(), 32, 32);

    // For all-positive input: grad should be all ones (pass-through)
    test::assert_all_close(x->grad.value(), 1.0f, 0.1f);

    return 0;
}
