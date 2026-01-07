// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Test: Basic tensor wrapper

#include "dynamic/autograd.hpp"
#include "common.hpp"
#include <ttnn/operations/matmul/matmul.hpp>

using namespace autograd;

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    // zeros tensor
    auto v1 = make_zeros(ttnn::Shape({32, 64}), device, false, "v1");
    test::assert_shape(v1->data, 32, 64);
    assert(!v1->requires_grad);

    // ones tensor with grad
    auto v2 = make_ones(ttnn::Shape({64, 32}), device, true, "v2");
    test::assert_shape(v2->data, 64, 32);
    assert(v2->requires_grad);

    // zero_grad allocates gradient
    assert(!v2->grad.has_value());
    v2->zero_grad(device);
    assert(v2->grad.has_value());

    // forward matmul [32,64] @ [64,32] = [32,32]
    auto a = make_ones(ttnn::Shape({32, 64}), device, false, "a");
    auto b = make_ones(ttnn::Shape({64, 32}), device, false, "b");
    auto c = from_tensor(ttnn::matmul(a->data, b->data), false, "c");
    test::assert_shape(c->data, 32, 32);

    // verify result: ones(32,64) @ ones(64,32) = 64 * ones(32,32)
    test::assert_all_close(c->data, 64.0f, 0.1f);

    return 0;
}
