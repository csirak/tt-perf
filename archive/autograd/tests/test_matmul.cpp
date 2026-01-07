// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Test: Matmul forward + backward

#include "dynamic/autograd.hpp"
#include "dynamic/ops.hpp"
#include "common.hpp"

using namespace autograd;

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    // A[32,64] @ B[64,32] = C[32,32]
    auto a = make_ones(ttnn::Shape({32, 64}), device, true, "a");
    auto b = make_ones(ttnn::Shape({64, 32}), device, true, "b");
    auto c = mm(a, b);

    // verify forward: each element = 64 (sum of 64 ones)
    test::assert_all_close(c->data, 64.0f, 0.1f);

    // backward
    backward(c, device);
    assert(a->grad.has_value() && b->grad.has_value());

    // dA = dC @ B.T = ones(32,32) @ ones(32,64) = 32 * ones(32,64)
    test::assert_shape(a->grad.value(), 32, 64);
    test::assert_all_close(a->grad.value(), 32.0f, 0.1f);

    // dB = A.T @ dC = ones(64,32) @ ones(32,32) = 32 * ones(64,32)
    test::assert_shape(b->grad.value(), 64, 32);
    test::assert_all_close(b->grad.value(), 32.0f, 0.1f);

    return 0;
}
