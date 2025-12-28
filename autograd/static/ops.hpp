// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Static Autograd Operations
// Each op writes to pre-allocated buffers and returns a Value* for the graph.
// Backward functions accumulate gradients into pre-allocated gradient buffers.

#pragma once

#include "value.hpp"
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>

namespace static_autograd {

// Matmul: out = a @ b
// Backward: da += dout @ b.T, db += a.T @ dout
inline Value* mm(Graph& g, Value* a, Value* b, Tensor* out, Tensor* d_out) {
    // Forward
    *out = ttnn::matmul(*a->data, *b->data);

    auto* v = g.node(out, d_out);
    v->parents = {a, b};

    if (!a->requires_grad && !b->requires_grad) {
        v->requires_grad = false;
        return v;
    }

    v->backward_fn = [a, b, v]() {
        if (!v->grad) return;
        const auto& dout = *v->grad;

        if (a->requires_grad) {
            a->accumulate_grad(ttnn::matmul(dout, *b->data, false, true));
        }
        if (b->requires_grad) {
            b->accumulate_grad(ttnn::matmul(*a->data, dout, true, false));
        }
    };
    return v;
}

// Add: out = a + b
// Backward: da += dout, db += dout (or sum if broadcast)
inline Value* add(Graph& g, Value* a, Value* b, Tensor* out, Tensor* d_out) {
    *out = ttnn::add(*a->data, *b->data);

    auto* v = g.node(out, d_out);
    v->parents = {a, b};

    if (!a->requires_grad && !b->requires_grad) {
        v->requires_grad = false;
        return v;
    }

    v->backward_fn = [a, b, v]() {
        if (!v->grad) return;
        const auto& dout = *v->grad;

        if (a->requires_grad) {
            a->accumulate_grad(dout);
        }
        if (b->requires_grad) {
            // Handle broadcast: sum if shapes differ
            if (b->data->logical_shape() != dout.logical_shape()) {
                b->accumulate_grad(ttnn::sum(dout, 0, true));
            } else {
                b->accumulate_grad(dout);
            }
        }
    };
    return v;
}

// Subtract: out = a - b
// Backward: da += dout, db -= dout
inline Value* sub(Graph& g, Value* a, Value* b, Tensor* out, Tensor* d_out) {
    *out = ttnn::subtract(*a->data, *b->data);

    auto* v = g.node(out, d_out);
    v->parents = {a, b};

    if (!a->requires_grad && !b->requires_grad) {
        v->requires_grad = false;
        return v;
    }

    v->backward_fn = [a, b, v]() {
        if (!v->grad) return;
        const auto& dout = *v->grad;

        if (a->requires_grad) {
            a->accumulate_grad(dout);
        }
        if (b->requires_grad) {
            b->accumulate_grad(ttnn::neg(dout));
        }
    };
    return v;
}

// Multiply (element-wise): out = a * b
// Backward: da += dout * b, db += dout * a
inline Value* mul(Graph& g, Value* a, Value* b, Tensor* out, Tensor* d_out) {
    *out = ttnn::multiply(*a->data, *b->data);

    auto* v = g.node(out, d_out);
    v->parents = {a, b};

    if (!a->requires_grad && !b->requires_grad) {
        v->requires_grad = false;
        return v;
    }

    v->backward_fn = [a, b, v]() {
        if (!v->grad) return;
        const auto& dout = *v->grad;

        if (a->requires_grad) {
            a->accumulate_grad(ttnn::multiply(dout, *b->data));
        }
        if (b->requires_grad) {
            b->accumulate_grad(ttnn::multiply(dout, *a->data));
        }
    };
    return v;
}

// ReLU: out = max(x, 0)
// Backward: dx += dout * (x > 0)
// Note: mask buffer stores (x > 0) for backward
inline Value* relu(Graph& g, Value* x, Tensor* out, Tensor* d_out, Tensor* mask) {
    *mask = ttnn::gtz(*x->data);
    *out = ttnn::relu(*x->data);

    auto* v = g.node(out, d_out);
    v->parents = {x};

    if (!x->requires_grad) {
        v->requires_grad = false;
        return v;
    }

    v->backward_fn = [x, v, mask]() {
        if (!v->grad) return;
        if (x->requires_grad) {
            x->accumulate_grad(ttnn::multiply(*v->grad, *mask));
        }
    };
    return v;
}

// Softmax: out = softmax(x, dim)
// Backward: dx += out * (dout - sum(dout * out, dim, keepdim))
inline Value* softmax(Graph& g, Value* x, int dim, Tensor* out, Tensor* d_out) {
    // Max-center for numerical stability
    auto x_max = ttnn::max(*x->data, dim, true);
    auto x_centered = ttnn::subtract(*x->data, x_max);
    *out = ttnn::softmax(x_centered, dim);

    auto* v = g.node(out, d_out);
    v->parents = {x};

    if (!x->requires_grad) {
        v->requires_grad = false;
        return v;
    }

    v->backward_fn = [x, v, out, dim]() {
        if (!v->grad) return;
        if (x->requires_grad) {
            const auto& dout = *v->grad;
            auto dy_y = ttnn::multiply(dout, *out);
            auto sum_dy_y = ttnn::sum(dy_y, dim, true);
            x->accumulate_grad(ttnn::multiply(*out, ttnn::subtract(dout, sum_dy_y)));
        }
    };
    return v;
}

// Mean: out = mean(x)
// Backward: dx += dout / numel broadcast to x shape
inline Value* mean(Graph& g, Value* x, Tensor* out, Tensor* d_out) {
    *out = ttnn::mean(*x->data, std::nullopt, true);

    auto* v = g.node(out, d_out);
    v->parents = {x};

    if (!x->requires_grad) {
        v->requires_grad = false;
        return v;
    }

    auto numel = x->data->logical_shape().volume();
    v->backward_fn = [x, v, numel]() {
        if (!v->grad) return;
        if (x->requires_grad) {
            float scale = 1.0f / static_cast<float>(numel);
            x->accumulate_grad(ttnn::multiply(ttnn::full_like(*x->data, scale), *v->grad));
        }
    };
    return v;
}

// MSE Loss: out = mean((pred - target)^2)
// Backward: d_pred += 2 * (pred - target) / numel
// Note: diff buffer stores (pred - target) for backward
inline Value* mse(Graph& g, Value* pred, Tensor* target, Tensor* out, Tensor* d_out, Tensor* diff) {
    *diff = ttnn::subtract(*pred->data, *target);
    auto sq = ttnn::multiply(*diff, *diff);
    *out = ttnn::mean(sq, std::nullopt, true);

    auto* v = g.node(out, d_out);
    v->parents = {pred};

    if (!pred->requires_grad) {
        v->requires_grad = false;
        return v;
    }

    auto numel = pred->data->logical_shape().volume();
    v->backward_fn = [pred, diff, numel]() {
        if (pred->requires_grad) {
            // d_pred = 2 * diff / numel
            // Note: v->grad is ones_like(loss) = scalar 1.0, so we skip multiplying by it
            float scale = 2.0f / static_cast<float>(numel);
            pred->accumulate_grad(ttnn::multiply(*diff, scale));
        }
    };
    return v;
}

}  // namespace static_autograd
