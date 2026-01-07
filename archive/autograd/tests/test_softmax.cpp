// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Test: Softmax forward + backward

#include "dynamic/autograd.hpp"
#include "dynamic/ops.hpp"
#include "common.hpp"

using namespace autograd;

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    // Create input [32, 64] with ones
    auto x = make_ones(ttnn::Shape({32, 64}), device, true, "x");

    // Forward: softmax(ones) = uniform distribution = 1/64 per element
    auto y = autograd::softmax(x, -1);
    test::assert_shape(y->data, 32, 64);
    assert(y->requires_grad);

    // Verify forward: each element should be 1/64 ~ 0.015625
    test::assert_all_close(y->data, 1.0f / 64.0f, 0.01f);

    // Backward
    backward(y, device);
    assert(x->grad.has_value());

    // For uniform softmax with uniform upstream gradient (all ones):
    // dX = Y * (dY - sum(dY * Y))
    // sum(dY * Y) = sum(1 * 1/64) = 1 per row
    // dX = (1/64) * (1 - 1) = 0
    test::assert_all_close(x->grad.value(), 0.0f, 0.01f);

    return 0;
}
