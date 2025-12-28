// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Traced Operations - Static ops with output tensor parameters
// All operations write to pre-allocated tensors for trace compatibility.

#pragma once

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/operations/matmul/matmul.hpp>
#include <ttnn/operations/eltwise/unary/unary.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/reduction/generic/generic_reductions.hpp>
#include <ttnn/operations/normalization/softmax/softmax.hpp>
#include <ttnn/operations/data_movement/transpose/transpose.hpp>
#include <ttnn/operations/eltwise/unary_backward/unary_backward.hpp>

namespace traced {

using Tensor = tt::tt_metal::Tensor;

// ============================================================================
// Forward Operations
// ============================================================================

inline void matmul(const Tensor& a, const Tensor& b, Tensor& out) {
    out = ttnn::matmul(a, b);
}

inline void matmul_transpose_a(const Tensor& a, const Tensor& b, Tensor& out) {
    out = ttnn::matmul(a, b, true, false);
}

inline void matmul_transpose_b(const Tensor& a, const Tensor& b, Tensor& out) {
    out = ttnn::matmul(a, b, false, true);
}

inline void add(const Tensor& a, const Tensor& b, Tensor& out) {
    out = ttnn::add(a, b);
}

inline void subtract(const Tensor& a, const Tensor& b, Tensor& out) {
    out = ttnn::subtract(a, b);
}

inline void multiply(const Tensor& a, const Tensor& b, Tensor& out) {
    out = ttnn::multiply(a, b);
}

inline void multiply_scalar(const Tensor& a, float s, Tensor& out) {
    out = ttnn::multiply(a, s);
}

inline void relu(const Tensor& x, Tensor& out, Tensor& mask) {
    mask = ttnn::gtz(x);  // Cache mask for backward
    out = ttnn::relu(x);
}

inline void gelu(const Tensor& x, Tensor& out) {
    out = ttnn::gelu(x);
}

inline void mean(const Tensor& x, Tensor& out) {
    out = ttnn::mean(x, std::nullopt, true);
}

inline void sum_dim0(const Tensor& x, Tensor& out) {
    out = ttnn::sum(x, 0, true);
}

inline void sum_dim(const Tensor& x, int dim, Tensor& out) {
    out = ttnn::sum(x, dim, true);
}

// Softmax with max-centering for numerical stability
// softmax(x) = softmax(x - max(x))
inline void softmax(const Tensor& x, int dim, Tensor& out) {
    auto x_max = ttnn::max(x, dim, true);
    auto x_centered = ttnn::subtract(x, x_max);
    out = ttnn::softmax(x_centered, dim);
}

// Transpose last two dimensions (for attention: K -> K.T)
inline void transpose_last2(const Tensor& x, Tensor& out) {
    out = ttnn::transpose(x, -2, -1);
}

// ============================================================================
// Backward Operations
// ============================================================================

inline void relu_backward(const Tensor& d_out, const Tensor& mask, Tensor& d_in) {
    d_in = ttnn::multiply(d_out, mask);
}

inline void gelu_backward(const Tensor& d_out, const Tensor& x, Tensor& d_in) {
    auto grads = ttnn::gelu_bw(d_out, x, "none");
    d_in = grads[0].value();
}

// Softmax backward: d_in = softmax * (d_out - sum(d_out * softmax, dim, keepdim=True))
inline void softmax_backward(const Tensor& d_out, const Tensor& softmax_out, int dim, Tensor& d_in) {
    auto dy_y = ttnn::multiply(d_out, softmax_out);
    auto sum_dy_y = ttnn::sum(dy_y, dim, true);
    d_in = ttnn::multiply(softmax_out, ttnn::subtract(d_out, sum_dy_y));
}

// Linear layer backward: computes d_weight, d_bias, and d_input
// y = x @ weight.T + bias
// d_weight = d_out.T @ x  (but weight is [out, in], so need d_out.T @ x)
// d_bias = sum(d_out, dim=0)
// d_input = d_out @ weight
inline void linear_backward(const Tensor& x, const Tensor& d_out,
                            const Tensor& weight,
                            Tensor& d_weight, Tensor& d_bias, Tensor& d_input) {
    d_weight = ttnn::matmul(d_out, x, true, false);
    d_bias = ttnn::sum(d_out, 0, true);
    d_input = ttnn::matmul(d_out, weight);
}

// MSE loss backward: d_pred = 2 * (pred - target) / numel
// But we split this: diff = pred - target computed in forward
// d_pred = diff * (2.0 * loss_scale)
inline void mse_backward(const Tensor& diff, float scale, Tensor& d_pred) {
    d_pred = ttnn::multiply(diff, 2.0f * scale);
}

// ============================================================================
// SGD Update
// ============================================================================

inline void sgd_update(Tensor& param, const Tensor& grad, float lr) {
    param = ttnn::subtract(param, ttnn::multiply(grad, lr));
}

}  // namespace traced
