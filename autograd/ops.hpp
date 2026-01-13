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
#include <ttnn/operations/eltwise/unary_backward/unary_backward.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>
#include "tensordiff/core/base_tensor.hpp"
#include "tensordiff/core/ttnn_bridge.hpp"
#include <cstdlib>
#include <string>
#include <string_view>
#include <initializer_list>
#include <vector>

namespace static_autograd {

inline bool tensordiff_recording_enabled() {
    return tensordiff::core::op_recording_enabled();
}

inline void record_ttnn_op(std::string_view op_name,
                           tensordiff::core::OpType type,
                           const Tensor& output,
                           std::initializer_list<const Tensor*> inputs,
                           std::string_view opts = {}) {
    if (!tensordiff_recording_enabled()) {
        return;
    }
    std::vector<tensordiff::core::CpuTensor> cpu_inputs;
    cpu_inputs.reserve(inputs.size());
    for (const auto* t : inputs) {
        cpu_inputs.push_back(tensordiff::core::from_ttnn(*t));
    }
    std::vector<const tensordiff::core::CpuTensor*> input_ptrs;
    input_ptrs.reserve(cpu_inputs.size());
    for (auto& t : cpu_inputs) {
        input_ptrs.push_back(&t);
    }
    std::vector<std::string_view> origins(input_ptrs.size(), "ttnn");
    auto out_cpu = tensordiff::core::from_ttnn(output);
    tensordiff::core::record_op_cpu(op_name, type, input_ptrs, origins, out_cpu, "ttnn", opts);
}

inline ttnn::WormholeComputeKernelConfig get_softmax_compute_config() {
    return ttnn::WormholeComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi2,
        .math_approx_mode = false,
        .fp32_dest_acc_en = true,
        .packer_l1_acc = true,
    };
}

inline const char* gelu_approx_mode() {
    const char* env = std::getenv("GELU_APPROX");
    return (env && *env) ? env : "none";
}

inline bool gelu_fast_mode() {
    return std::string_view(gelu_approx_mode()) == "tanh";
}

inline Tensor gelu_forward_tensor(const Tensor& x) {
    return ttnn::gelu(x, gelu_fast_mode());
}

// Matmul: out = a @ b
// Backward: da += dout @ b.T, db += a.T @ dout
inline Value* mm(Graph& g, Value* a, Value* b, Tensor* out, Tensor* d_out) {
    // Forward
    *out = ttnn::matmul(*a->data, *b->data);
    record_ttnn_op("mm", tensordiff::core::OpType::Binary, *out, {a->data, b->data}, "ta=0,tb=0");

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
            auto da = ttnn::matmul(dout, *b->data, false, true);
            record_ttnn_op("mm_bw_da", tensordiff::core::OpType::Binary, da, {&dout, b->data}, "ta=0,tb=1");
            a->accumulate_grad(da);
        }
        if (b->requires_grad) {
            auto db = ttnn::matmul(*a->data, dout, true, false);
            record_ttnn_op("mm_bw_db", tensordiff::core::OpType::Binary, db, {a->data, &dout}, "ta=1,tb=0");
            b->accumulate_grad(db);
        }
    };
    return v;
}

// Add: out = a + b
// Backward: da += dout, db += dout (or sum if broadcast)
inline Value* add(Graph& g, Value* a, Value* b, Tensor* out, Tensor* d_out) {
    *out = ttnn::add(*a->data, *b->data);
    record_ttnn_op("add", tensordiff::core::OpType::Binary, *out, {a->data, b->data});

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
            record_ttnn_op("add_bw_da", tensordiff::core::OpType::Unary, dout, {&dout});
            a->accumulate_grad(dout);
        }
        if (b->requires_grad) {
            // Handle broadcast: sum if shapes differ
            if (b->data->logical_shape() != dout.logical_shape()) {
                auto db = ttnn::sum(dout, 0, true);
                record_ttnn_op("add_bw_db_sum", tensordiff::core::OpType::Unary, db, {&dout}, "axis=0,keepdim=1");
                b->accumulate_grad(db);
            } else {
                record_ttnn_op("add_bw_db", tensordiff::core::OpType::Unary, dout, {&dout});
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
    record_ttnn_op("sub", tensordiff::core::OpType::Binary, *out, {a->data, b->data});

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
            record_ttnn_op("sub_bw_da", tensordiff::core::OpType::Unary, dout, {&dout});
            a->accumulate_grad(dout);
        }
        if (b->requires_grad) {
            auto db = ttnn::neg(dout);
            record_ttnn_op("sub_bw_db", tensordiff::core::OpType::Unary, db, {&dout});
            b->accumulate_grad(db);
        }
    };
    return v;
}

// Multiply (element-wise): out = a * b
// Backward: da += dout * b, db += dout * a
inline Value* mul(Graph& g, Value* a, Value* b, Tensor* out, Tensor* d_out) {
    *out = ttnn::multiply(*a->data, *b->data);
    record_ttnn_op("mul", tensordiff::core::OpType::Binary, *out, {a->data, b->data});

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
            auto da = ttnn::multiply(dout, *b->data);
            record_ttnn_op("mul_bw_da", tensordiff::core::OpType::Binary, da, {&dout, b->data});
            a->accumulate_grad(da);
        }
        if (b->requires_grad) {
            auto db = ttnn::multiply(dout, *a->data);
            record_ttnn_op("mul_bw_db", tensordiff::core::OpType::Binary, db, {&dout, a->data});
            b->accumulate_grad(db);
        }
    };
    return v;
}

// ReLU: out = max(x, 0)
// Backward: dx += dout * (x > 0)
// Note: mask buffer stores (x > 0) for backward
inline Value* relu(Graph& g, Value* x, Tensor* out, Tensor* d_out, Tensor* mask) {
    *mask = ttnn::gtz(*x->data);
    record_ttnn_op("gtz", tensordiff::core::OpType::Unary, *mask, {x->data});
    *out = ttnn::relu(*x->data);
    record_ttnn_op("relu", tensordiff::core::OpType::Unary, *out, {x->data});

    auto* v = g.node(out, d_out);
    v->parents = {x};

    if (!x->requires_grad) {
        v->requires_grad = false;
        return v;
    }

    v->backward_fn = [x, v, mask]() {
        if (!v->grad) return;
        if (x->requires_grad) {
            auto dx = ttnn::multiply(*v->grad, *mask);
            record_ttnn_op("relu_bw", tensordiff::core::OpType::Binary, dx, {v->grad, mask});
            x->accumulate_grad(dx);
        }
    };
    return v;
}

// GELU: out = gelu(x)
// Backward: uses ttnn::gelu_bw
inline Value* gelu(Graph& g, Value* x, Tensor* out, Tensor* d_out) {
    *out = gelu_forward_tensor(*x->data);
    record_ttnn_op("gelu", tensordiff::core::OpType::Unary, *out, {x->data},
                   std::string("approx=") + gelu_approx_mode());

    auto* v = g.node(out, d_out);
    v->parents = {x};

    if (!x->requires_grad) {
        v->requires_grad = false;
        return v;
    }

    v->backward_fn = [x, v]() {
        if (!v->grad) return;
        if (x->requires_grad) {
            auto grads = ttnn::gelu_bw(*v->grad, *x->data, gelu_approx_mode());
            auto dx = grads[0].value();
            record_ttnn_op("gelu_bw", tensordiff::core::OpType::Binary, dx, {v->grad, x->data},
                           std::string("approx=") + gelu_approx_mode());
            x->accumulate_grad(dx);
        }
    };
    return v;
}

// Softmax: out = softmax(x, dim)
// Backward: dx += out * (dout - sum(dout * out, dim, keepdim))
inline Value* softmax(Graph& g, Value* x, int dim, Tensor* out, Tensor* d_out) {
    *out = ttnn::softmax(*x->data, dim, std::nullopt, get_softmax_compute_config(), true);
    record_ttnn_op("softmax", tensordiff::core::OpType::Unary, *out, {x->data},
                   "dim=" + std::to_string(dim));

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
            record_ttnn_op("softmax_bw_mul1", tensordiff::core::OpType::Binary, dy_y, {&dout, out});
            auto sum_dy_y = ttnn::sum(dy_y, dim, true);
            record_ttnn_op("softmax_bw_sum", tensordiff::core::OpType::Unary, sum_dy_y, {&dy_y},
                           "dim=" + std::to_string(dim) + ",keepdim=1");
            auto diff = ttnn::subtract(dout, sum_dy_y);
            record_ttnn_op("softmax_bw_sub", tensordiff::core::OpType::Binary, diff, {&dout, &sum_dy_y});
            auto dx = ttnn::multiply(*out, diff);
            record_ttnn_op("softmax_bw_mul2", tensordiff::core::OpType::Binary, dx, {out, &diff});
            x->accumulate_grad(dx);
        }
    };
    return v;
}

// Mean: out = mean(x)
// Backward: dx += dout / numel broadcast to x shape
inline Value* mean(Graph& g, Value* x, Tensor* out, Tensor* d_out) {
    *out = ttnn::mean(*x->data, std::nullopt, true);
    record_ttnn_op("mean", tensordiff::core::OpType::Unary, *out, {x->data}, "keepdim=1");

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
            auto scale_tensor = ttnn::full_like(*x->data, scale);
            record_ttnn_op("mean_bw_scale", tensordiff::core::OpType::Unary, scale_tensor, {x->data},
                           "scale=" + std::to_string(scale));
            auto dx = ttnn::multiply(scale_tensor, *v->grad);
            record_ttnn_op("mean_bw_mul", tensordiff::core::OpType::Binary, dx, {&scale_tensor, v->grad});
            x->accumulate_grad(dx);
        }
    };
    return v;
}

// MSE Loss: out = mean((pred - target)^2)
// Backward: d_pred += 2 * (pred - target) / numel
// Note: diff buffer stores (pred - target) for backward
inline Value* mse(Graph& g, Value* pred, Tensor* target, Tensor* out, Tensor* d_out, Tensor* diff) {
    *diff = ttnn::subtract(*pred->data, *target);
    record_ttnn_op("mse_sub", tensordiff::core::OpType::Binary, *diff, {pred->data, target});
    auto sq = ttnn::multiply(*diff, *diff);
    record_ttnn_op("mse_mul", tensordiff::core::OpType::Binary, sq, {diff, diff});
    *out = ttnn::mean(sq, std::nullopt, true);
    record_ttnn_op("mse_mean", tensordiff::core::OpType::Unary, *out, {&sq}, "keepdim=1");

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
            auto dx = ttnn::multiply(*diff, scale);
            record_ttnn_op("mse_bw_mul", tensordiff::core::OpType::Unary, dx, {diff},
                           "scale=" + std::to_string(scale));
            pred->accumulate_grad(dx);
        }
    };
    return v;
}

}  // namespace static_autograd
