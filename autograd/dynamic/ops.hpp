// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Autograd Operations for TTNN
// Each op: forward computation + backward_fn for gradient

#pragma once

#include "autograd.hpp"
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>

namespace autograd {

// Matmul: C = A @ B
// Backward: dA = dC @ B.T, dB = A.T @ dC
inline ValuePtr mm(ValuePtr a, ValuePtr b) {
    auto c_data = ttnn::matmul(a->data, b->data);
    auto c = std::make_shared<Value>(
        std::move(c_data),
        a->requires_grad || b->requires_grad,
        "matmul_out"
    );

    if (!c->requires_grad) return c;

    c->parents = {a, b};
    c->backward_fn = [a, b, c]() {
        if (!c->grad.has_value()) return;
        const auto& dc = c->grad.value();

        // dA = dC @ B.T  (use transpose_b flag instead of explicit transpose)
        if (a->requires_grad) {
            a->accumulate_grad(ttnn::matmul(dc, b->data, false, true));
        }
        // dB = A.T @ dC  (use transpose_a flag instead of explicit transpose)
        if (b->requires_grad) {
            b->accumulate_grad(ttnn::matmul(a->data, dc, true, false));
        }
    };
    return c;
}

// Add: Z = X + Y (with broadcasting)
// Backward: dX = dZ, dY = sum(dZ) if broadcast
inline ValuePtr add(ValuePtr a, ValuePtr b) {
    auto z = std::make_shared<Value>(
        ttnn::add(a->data, b->data),
        a->requires_grad || b->requires_grad,
        "add_out"
    );
    if (!z->requires_grad) return z;

    z->parents = {a, b};
    z->backward_fn = [a, b, z]() {
        if (!z->grad.has_value()) return;
        const auto& dz = z->grad.value();
        if (a->requires_grad) a->accumulate_grad(dz);
        if (b->requires_grad) {
            // Handle broadcast: sum over batch dimension if shapes differ
            if (b->data.logical_shape() != dz.logical_shape())
                b->accumulate_grad(ttnn::sum(dz, 0, false));
            else
                b->accumulate_grad(dz);
        }
    };
    return z;
}

// Subtract: Z = X - Y
// Backward: dX = dZ, dY = -dZ
inline ValuePtr sub(ValuePtr a, ValuePtr b) {
    auto z = std::make_shared<Value>(
        ttnn::subtract(a->data, b->data),
        a->requires_grad || b->requires_grad,
        "sub_out"
    );
    if (!z->requires_grad) return z;

    z->parents = {a, b};
    z->backward_fn = [a, b, z]() {
        if (!z->grad.has_value()) return;
        const auto& dz = z->grad.value();
        if (a->requires_grad) a->accumulate_grad(dz);
        if (b->requires_grad) b->accumulate_grad(ttnn::neg(dz));
    };
    return z;
}

// Multiply (element-wise): Z = X * Y
// Backward: dX = dZ * Y, dY = dZ * X
inline ValuePtr mul(ValuePtr a, ValuePtr b) {
    auto z = std::make_shared<Value>(
        ttnn::multiply(a->data, b->data),
        a->requires_grad || b->requires_grad,
        "mul_out"
    );
    if (!z->requires_grad) return z;

    z->parents = {a, b};
    z->backward_fn = [a, b, z]() {
        if (!z->grad.has_value()) return;
        const auto& dz = z->grad.value();
        if (a->requires_grad) a->accumulate_grad(ttnn::multiply(dz, b->data));
        if (b->requires_grad) b->accumulate_grad(ttnn::multiply(dz, a->data));
    };
    return z;
}

// Mean: Y = mean(X) -> scalar
// Backward: dX = dY / numel(X) broadcast to input shape
inline ValuePtr reduce_mean(ValuePtr x) {
    auto numel = x->data.logical_shape().volume();
    // Use keepdim=true to preserve rank for easier broadcasting in backward
    auto y = std::make_shared<Value>(ttnn::mean(x->data, std::nullopt, true), x->requires_grad, "mean_out");
    if (!x->requires_grad) return y;

    y->parents = {x};
    y->backward_fn = [x, y, numel]() {
        if (!y->grad.has_value()) return;
        // dX = dY / N - broadcast dy to x shape and scale by 1/N
        float scale = 1.0f / static_cast<float>(numel);
        // Use multiply to broadcast dy (scalar or 1x1) to x shape, then scale
        auto dx = ttnn::multiply(ttnn::full_like(x->data, scale), y->grad.value());
        x->accumulate_grad(dx);
    };
    return y;
}

// ReLU: Y = max(X, 0)
// Backward: dX = dY * (X > 0)
inline ValuePtr relu(ValuePtr x) {
    auto y = std::make_shared<Value>(ttnn::relu(x->data), x->requires_grad);
    if (!x->requires_grad) return y;

    y->parents = {x};
    y->backward_fn = [x, y]() {
        if (!y->grad.has_value()) return;
        x->accumulate_grad(ttnn::multiply(y->grad.value(), ttnn::gtz(x->data)));
    };
    return y;
}

// Softmax: Y = softmax(X - max(X), dim) for numerical stability
// Backward: dX = Y * (dY - sum(dY * Y, dim, keepdim=True))
inline ValuePtr softmax(ValuePtr x, int dim = -1) {
    // Max-center for numerical stability: softmax(x) = softmax(x - max(x))
    auto x_max = ttnn::max(x->data, dim, true);
    auto x_centered = ttnn::subtract(x->data, x_max);
    auto y = std::make_shared<Value>(ttnn::softmax(x_centered, dim), x->requires_grad);
    if (!x->requires_grad) return y;

    y->parents = {x};
    y->backward_fn = [x, y, dim]() {
        if (!y->grad.has_value()) return;
        const auto& dy = y->grad.value();
        auto dy_y = ttnn::multiply(dy, y->data);
        auto sum_dy_y = ttnn::sum(dy_y, dim, true);
        x->accumulate_grad(ttnn::multiply(y->data, ttnn::subtract(dy, sum_dy_y)));
    };
    return y;
}

// MSE Loss: Y = mean((pred - target)^2)
inline ValuePtr mse_loss(ValuePtr pred, ValuePtr target) {
    auto diff = sub(pred, target);
    auto sq = mul(diff, diff);
    return reduce_mean(sq);
}

}  // namespace autograd
