// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "autograd.hpp"

#include "tensordiff/core/compare.hpp"
#include "tensordiff/core/base_tensor.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tensordiffgrad::core {

using tensordiff::core::record_op_cpu;
using tensordiff::core::OpType;
using tensordiff::core::to_float_vec;

inline void record_named(std::string_view tag,
                         OpType type,
                         const std::vector<Value*>& inputs,
                         Value* output,
                         std::string_view opts = {}) {
    std::string op_name;
    std::string_view op_view = tag;
    if (op_view.empty() && output && !output->name.empty()) {
        op_name = output->name;
        op_view = op_name;
    }
    if (op_view.empty()) return;
    std::vector<const CpuTensor*> tensors;
    std::vector<std::string_view> origins;
    std::vector<std::string_view> names;
    tensors.reserve(inputs.size());
    origins.reserve(inputs.size());
    names.reserve(inputs.size());
    for (auto* v : inputs) {
        tensors.push_back(&v->data);
        origins.push_back("cpu");
        names.push_back(v ? v->name : std::string_view());
    }
    const std::string_view out_name = output ? std::string_view(output->name) : std::string_view();
    record_op_cpu(std::string(op_view), type, tensors, origins, names, output->data, "cpu", out_name, opts);
}

inline void ensure_same_shape(const CpuTensor& a, const CpuTensor& b, std::string_view name) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error(std::string(name) + ": shape mismatch");
    }
}

inline CpuTensor add_tensors(const CpuTensor& a, const CpuTensor& b) {
    ensure_same_shape(a, b, "add");
    auto af = to_float_vec(a);
    auto bf = to_float_vec(b);
    std::vector<float> out(af.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = af[i] + bf[i];
    }
    return CpuTensor::from_f32(std::move(out), a.shape());
}

inline CpuTensor sub_tensors(const CpuTensor& a, const CpuTensor& b) {
    ensure_same_shape(a, b, "sub");
    auto af = to_float_vec(a);
    auto bf = to_float_vec(b);
    std::vector<float> out(af.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = af[i] - bf[i];
    }
    return CpuTensor::from_f32(std::move(out), a.shape());
}

inline CpuTensor mul_tensors(const CpuTensor& a, const CpuTensor& b) {
    ensure_same_shape(a, b, "mul");
    auto af = to_float_vec(a);
    auto bf = to_float_vec(b);
    std::vector<float> out(af.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = af[i] * bf[i];
    }
    return CpuTensor::from_f32(std::move(out), a.shape());
}

inline CpuTensor mul_scalar(const CpuTensor& a, float scalar) {
    auto af = to_float_vec(a);
    std::vector<float> out(af.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = af[i] * scalar;
    }
    return CpuTensor::from_f32(std::move(out), a.shape());
}

inline CpuTensor add_scalar(const CpuTensor& a, float scalar) {
    auto af = to_float_vec(a);
    std::vector<float> out(af.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = af[i] + scalar;
    }
    return CpuTensor::from_f32(std::move(out), a.shape());
}

inline CpuTensor broadcast_1x1d_to_bsd(const CpuTensor& v, uint32_t bsz, uint32_t seq, uint32_t dim) {
    auto vf = to_float_vec(v);
    if (vf.size() != dim) {
        throw std::runtime_error("broadcast_1x1d_to_bsd: dim mismatch");
    }
    std::vector<float> out(static_cast<size_t>(bsz) * seq * dim);
    for (uint32_t b0 = 0; b0 < bsz; ++b0) {
        for (uint32_t s = 0; s < seq; ++s) {
            const size_t base = (static_cast<size_t>(b0) * seq + s) * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                out[base + d] = vf[d];
            }
        }
    }
    return CpuTensor::from_f32(std::move(out), {bsz, seq, dim});
}

inline CpuTensor matmul_2d(const CpuTensor& a, const CpuTensor& b, uint32_t m, uint32_t k, uint32_t n) {
    auto af = to_float_vec(a);
    auto bf = to_float_vec(b);
    std::vector<float> out(static_cast<size_t>(m) * n, 0.0f);
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (uint32_t kk = 0; kk < k; ++kk) {
                acc += af[static_cast<size_t>(i) * k + kk] * bf[static_cast<size_t>(kk) * n + j];
            }
            out[static_cast<size_t>(i) * n + j] = acc;
        }
    }
    return CpuTensor::from_f32(std::move(out), {m, n});
}

inline CpuTensor matmul_3d_2d(const CpuTensor& a, const CpuTensor& b, uint32_t bsz, uint32_t seq, uint32_t k, uint32_t n) {
    // a: [B,S,K], b: [K,N] => out [B,S,N]
    auto af = to_float_vec(a);
    auto bf = to_float_vec(b);
    std::vector<float> out(static_cast<size_t>(bsz) * seq * n, 0.0f);
    const size_t row_stride = static_cast<size_t>(k);
    const size_t out_stride = static_cast<size_t>(n);
    for (uint32_t b0 = 0; b0 < bsz; ++b0) {
        for (uint32_t s = 0; s < seq; ++s) {
            const size_t a_base = (static_cast<size_t>(b0) * seq + s) * row_stride;
            const size_t o_base = (static_cast<size_t>(b0) * seq + s) * out_stride;
            for (uint32_t j = 0; j < n; ++j) {
                float acc = 0.0f;
                for (uint32_t kk = 0; kk < k; ++kk) {
                    acc += af[a_base + kk] * bf[static_cast<size_t>(kk) * n + j];
                }
                out[o_base + j] = acc;
            }
        }
    }
    return CpuTensor::from_f32(std::move(out), {bsz, seq, n});
}

inline CpuTensor matmul_batched_qk(const CpuTensor& q, const CpuTensor& k, uint32_t bsz, uint32_t seq, uint32_t d) {
    // q: [B,S,D], k: [B,S,D] => scores [B,S,S] using k^T
    auto qf = to_float_vec(q);
    auto kf = to_float_vec(k);
    std::vector<float> out(static_cast<size_t>(bsz) * seq * seq, 0.0f);
    for (uint32_t b0 = 0; b0 < bsz; ++b0) {
        for (uint32_t i = 0; i < seq; ++i) {
            for (uint32_t j = 0; j < seq; ++j) {
                float acc = 0.0f;
                const size_t q_base = (static_cast<size_t>(b0) * seq + i) * d;
                const size_t k_base = (static_cast<size_t>(b0) * seq + j) * d;
                for (uint32_t kk = 0; kk < d; ++kk) {
                    acc += qf[q_base + kk] * kf[k_base + kk];
                }
                out[(static_cast<size_t>(b0) * seq + i) * seq + j] = acc;
            }
        }
    }
    return CpuTensor::from_f32(std::move(out), {bsz, seq, seq});
}

inline CpuTensor matmul_batched_av(const CpuTensor& a, const CpuTensor& v, uint32_t bsz, uint32_t seq, uint32_t d) {
    // a: [B,S,S], v: [B,S,D] => out [B,S,D]
    auto af = to_float_vec(a);
    auto vf = to_float_vec(v);
    std::vector<float> out(static_cast<size_t>(bsz) * seq * d, 0.0f);
    for (uint32_t b0 = 0; b0 < bsz; ++b0) {
        for (uint32_t i = 0; i < seq; ++i) {
            for (uint32_t j = 0; j < d; ++j) {
                float acc = 0.0f;
                for (uint32_t kidx = 0; kidx < seq; ++kidx) {
                    const size_t a_idx = (static_cast<size_t>(b0) * seq + i) * seq + kidx;
                    const size_t v_idx = (static_cast<size_t>(b0) * seq + kidx) * d + j;
                    acc += af[a_idx] * vf[v_idx];
                }
                out[(static_cast<size_t>(b0) * seq + i) * d + j] = acc;
            }
        }
    }
    return CpuTensor::from_f32(std::move(out), {bsz, seq, d});
}

inline CpuTensor softmax_lastdim(const CpuTensor& x, uint32_t bsz, uint32_t seq, uint32_t dim) {
    auto xf = to_float_vec(x);
    std::vector<float> out(xf.size());
    for (uint32_t b0 = 0; b0 < bsz; ++b0) {
        for (uint32_t i = 0; i < seq; ++i) {
            const size_t base = (static_cast<size_t>(b0) * seq + i) * dim;
            float maxv = xf[base];
            for (uint32_t j = 0; j < dim; ++j) {
                maxv = std::max(maxv, xf[base + j]);
            }
            float sum = 0.0f;
            for (uint32_t j = 0; j < dim; ++j) {
                float e = std::exp(xf[base + j] - maxv);
                out[base + j] = e;
                sum += e;
            }
            float inv = 1.0f / sum;
            for (uint32_t j = 0; j < dim; ++j) {
                out[base + j] *= inv;
            }
        }
    }
    return CpuTensor::from_f32(std::move(out), {bsz, seq, dim});
}

inline CpuTensor gelu_tanh(const CpuTensor& x) {
    auto xf = to_float_vec(x);
    std::vector<float> out(xf.size());
    const float k0 = 0.044715f;
    const float k1 = std::sqrt(2.0f / static_cast<float>(M_PI));
    for (size_t i = 0; i < xf.size(); ++i) {
        float v = xf[i];
        float u = k1 * (v + k0 * v * v * v);
        out[i] = 0.5f * v * (1.0f + std::tanh(u));
    }
    return CpuTensor::from_f32(std::move(out), x.shape());
}

inline CpuTensor gelu_tanh_backward(const CpuTensor& x, const CpuTensor& dout) {
    auto xf = to_float_vec(x);
    auto df = to_float_vec(dout);
    std::vector<float> out(xf.size());
    const float k0 = 0.044715f;
    const float k1 = std::sqrt(2.0f / static_cast<float>(M_PI));
    for (size_t i = 0; i < xf.size(); ++i) {
        float v = xf[i];
        float u = k1 * (v + k0 * v * v * v);
        float t = std::tanh(u);
        float sech2 = 1.0f - t * t;
        float du = k1 * (1.0f + 3.0f * k0 * v * v);
        float dy = 0.5f * (1.0f + t) + 0.5f * v * sech2 * du;
        out[i] = df[i] * dy;
    }
    return CpuTensor::from_f32(std::move(out), x.shape());
}

// Autograd ops
inline Value* add(Graph& g, Value* a, Value* b, std::string_view tag = {}) {
    auto out = add_tensors(a->data, b->data);
    auto* v = g.node(out, a->requires_grad || b->requires_grad, std::string(tag));
    v->parents = {a, b};
    v->backward_fn = [a, b, v]() {
        if (!v->grad) return;
        if (a->requires_grad) {
            a->accumulate_grad(*v->grad);
        }
        if (b->requires_grad) {
            b->accumulate_grad(*v->grad);
        }
    };
    record_named(tag, OpType::Binary, {a, b}, v);
    return v;
}

inline Value* sub(Graph& g, Value* a, Value* b, std::string_view tag = {}) {
    auto out = sub_tensors(a->data, b->data);
    auto* v = g.node(out, a->requires_grad || b->requires_grad, std::string(tag));
    v->parents = {a, b};
    v->backward_fn = [a, b, v]() {
        if (!v->grad) return;
        if (a->requires_grad) {
            a->accumulate_grad(*v->grad);
        }
        if (b->requires_grad) {
            auto nf = to_float_vec(*v->grad);
            for (auto& x : nf) x = -x;
            b->accumulate_grad(CpuTensor::from_f32(std::move(nf), b->data.shape()));
        }
    };
    record_named(tag, OpType::Binary, {a, b}, v);
    return v;
}

inline Value* mul(Graph& g, Value* a, Value* b, std::string_view tag = {}) {
    auto out = mul_tensors(a->data, b->data);
    auto* v = g.node(out, a->requires_grad || b->requires_grad, std::string(tag));
    v->parents = {a, b};
    v->backward_fn = [a, b, v]() {
        if (!v->grad) return;
        if (a->requires_grad) {
            a->accumulate_grad(mul_tensors(*v->grad, b->data));
        }
        if (b->requires_grad) {
            b->accumulate_grad(mul_tensors(*v->grad, a->data));
        }
    };
    record_named(tag, OpType::Binary, {a, b}, v);
    return v;
}

inline Value* add_bias(Graph& g,
                       Value* x,
                       Value* b,
                       uint32_t bsz,
                       uint32_t seq,
                       uint32_t dim,
                       std::string_view tag = {}) {
    auto bcast = broadcast_1x1d_to_bsd(b->data, bsz, seq, dim);
    auto out = add_tensors(x->data, bcast);
    auto* v = g.node(out, x->requires_grad || b->requires_grad, std::string(tag));
    v->parents = {x, b};
    v->backward_fn = [x, b, v, bsz, seq, dim]() {
        if (!v->grad) return;
        if (x->requires_grad) {
            x->accumulate_grad(*v->grad);
        }
        if (b->requires_grad) {
            auto df = to_float_vec(*v->grad);
            std::vector<float> db(dim, 0.0f);
            for (uint32_t b0 = 0; b0 < bsz; ++b0) {
                for (uint32_t s = 0; s < seq; ++s) {
                    const size_t base = (static_cast<size_t>(b0) * seq + s) * dim;
                    for (uint32_t d = 0; d < dim; ++d) {
                        db[d] += df[base + d];
                    }
                }
            }
            b->accumulate_grad(CpuTensor::from_f32(std::move(db), {1, 1, dim}));
        }
    };
    record_named(tag, OpType::Binary, {x, b}, v);
    return v;
}

inline Value* matmul(Graph& g, Value* a, Value* b, uint32_t m, uint32_t k, uint32_t n, std::string_view tag = {}) {
    auto out = matmul_2d(a->data, b->data, m, k, n);
    auto* v = g.node(out, a->requires_grad || b->requires_grad, std::string(tag));
    v->parents = {a, b};
    v->backward_fn = [a, b, v, m, k, n]() {
        if (!v->grad) return;
        const auto& dout = *v->grad;
        if (a->requires_grad) {
            // dA = dOut * B^T
            auto bf = to_float_vec(b->data);
            auto df = to_float_vec(dout);
            std::vector<float> out_a(static_cast<size_t>(m) * k, 0.0f);
            for (uint32_t i = 0; i < m; ++i) {
                for (uint32_t kk = 0; kk < k; ++kk) {
                    float acc = 0.0f;
                    for (uint32_t j = 0; j < n; ++j) {
                        acc += df[static_cast<size_t>(i) * n + j] * bf[static_cast<size_t>(kk) * n + j];
                    }
                    out_a[static_cast<size_t>(i) * k + kk] = acc;
                }
            }
            a->accumulate_grad(CpuTensor::from_f32(std::move(out_a), {m, k}));
        }
        if (b->requires_grad) {
            // dB = A^T * dOut
            auto af = to_float_vec(a->data);
            auto df = to_float_vec(dout);
            std::vector<float> out_b(static_cast<size_t>(k) * n, 0.0f);
            for (uint32_t kk = 0; kk < k; ++kk) {
                for (uint32_t j = 0; j < n; ++j) {
                    float acc = 0.0f;
                    for (uint32_t i = 0; i < m; ++i) {
                        acc += af[static_cast<size_t>(i) * k + kk] * df[static_cast<size_t>(i) * n + j];
                    }
                    out_b[static_cast<size_t>(kk) * n + j] = acc;
                }
            }
            b->accumulate_grad(CpuTensor::from_f32(std::move(out_b), {k, n}));
        }
    };
    record_named(tag, OpType::Binary, {a, b}, v);
    return v;
}

inline Value* matmul_3d_2d_op(Graph& g, Value* a, Value* b, uint32_t bsz, uint32_t seq, uint32_t k, uint32_t n, std::string_view tag = {}) {
    auto out = matmul_3d_2d(a->data, b->data, bsz, seq, k, n);
    auto* v = g.node(out, a->requires_grad || b->requires_grad, std::string(tag));
    v->parents = {a, b};
    v->backward_fn = [a, b, v, bsz, seq, k, n]() {
        if (!v->grad) return;
        const auto& dout = *v->grad;
        auto df = to_float_vec(dout);
        if (a->requires_grad) {
            // dA: [B,S,K] = dOut * B^T
            auto bf = to_float_vec(b->data);
            std::vector<float> out_a(static_cast<size_t>(bsz) * seq * k, 0.0f);
            for (uint32_t b0 = 0; b0 < bsz; ++b0) {
                for (uint32_t s = 0; s < seq; ++s) {
                    const size_t d_base = (static_cast<size_t>(b0) * seq + s) * n;
                    const size_t a_base = (static_cast<size_t>(b0) * seq + s) * k;
                    for (uint32_t kk = 0; kk < k; ++kk) {
                        float acc = 0.0f;
                        for (uint32_t j = 0; j < n; ++j) {
                            acc += df[d_base + j] * bf[static_cast<size_t>(kk) * n + j];
                        }
                        out_a[a_base + kk] = acc;
                    }
                }
            }
            a->accumulate_grad(CpuTensor::from_f32(std::move(out_a), {bsz, seq, k}));
        }
        if (b->requires_grad) {
            // dB: [K,N] = sum over B,S of A^T * dOut
            auto af = to_float_vec(a->data);
            std::vector<float> out_b(static_cast<size_t>(k) * n, 0.0f);
            for (uint32_t kk = 0; kk < k; ++kk) {
                for (uint32_t j = 0; j < n; ++j) {
                    float acc = 0.0f;
                    for (uint32_t b0 = 0; b0 < bsz; ++b0) {
                        for (uint32_t s = 0; s < seq; ++s) {
                            const size_t a_base = (static_cast<size_t>(b0) * seq + s) * k;
                            const size_t d_base = (static_cast<size_t>(b0) * seq + s) * n;
                            acc += af[a_base + kk] * df[d_base + j];
                        }
                    }
                    out_b[static_cast<size_t>(kk) * n + j] = acc;
                }
            }
            b->accumulate_grad(CpuTensor::from_f32(std::move(out_b), {k, n}));
        }
    };
    record_named(tag, OpType::Binary, {a, b}, v);
    return v;
}

inline Value* softmax(Graph& g, Value* x, uint32_t bsz, uint32_t seq, uint32_t dim, std::string_view tag = {}) {
    auto out = softmax_lastdim(x->data, bsz, seq, dim);
    auto* v = g.node(out, x->requires_grad, std::string(tag));
    v->parents = {x};
    v->backward_fn = [x, v, bsz, seq, dim]() {
        if (!v->grad) return;
        auto y = to_float_vec(v->data);
        auto dout = to_float_vec(*v->grad);
        std::vector<float> dx(dout.size());
        for (uint32_t b0 = 0; b0 < bsz; ++b0) {
            for (uint32_t i = 0; i < seq; ++i) {
                const size_t base = (static_cast<size_t>(b0) * seq + i) * dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < dim; ++j) {
                    dot += dout[base + j] * y[base + j];
                }
                for (uint32_t j = 0; j < dim; ++j) {
                    dx[base + j] = y[base + j] * (dout[base + j] - dot);
                }
            }
        }
        x->accumulate_grad(CpuTensor::from_f32(std::move(dx), x->data.shape()));
    };
    record_named(tag, OpType::Unary, {x}, v);
    return v;
}

inline Value* gelu(Graph& g, Value* x, std::string_view tag = {}) {
    auto out = gelu_tanh(x->data);
    auto* v = g.node(out, x->requires_grad, std::string(tag));
    v->parents = {x};
    v->backward_fn = [x, v]() {
        if (!v->grad) return;
        auto dx = gelu_tanh_backward(x->data, *v->grad);
        x->accumulate_grad(dx);
    };
    record_named(tag, OpType::Unary, {x}, v);
    return v;
}

inline Value* scale(Graph& g, Value* x, float scalar, std::string_view tag = {}) {
    auto out = mul_scalar(x->data, scalar);
    auto* v = g.node(out, x->requires_grad, std::string(tag));
    v->parents = {x};
    v->backward_fn = [x, v, scalar]() {
        if (!v->grad) return;
        x->accumulate_grad(mul_scalar(*v->grad, scalar));
    };
    record_named(tag, OpType::Unary, {x}, v, std::string("scale=") + std::to_string(scalar));
    return v;
}

inline Value* sum_all(Graph& g, Value* x, std::string_view tag = {}) {
    auto xf = to_float_vec(x->data);
    float sum = 0.0f;
    for (float v : xf) sum += v;
    CpuTensor out({1}, DType::kF32);
    auto* dst = reinterpret_cast<float*>(out.raw().data());
    dst[0] = sum;
    auto* v = g.node(out, x->requires_grad, std::string(tag));
    v->parents = {x};
    v->backward_fn = [x, v]() {
        if (!v->grad) return;
        auto gval = to_float_vec(*v->grad)[0];
        std::vector<float> out(x->data.numel(), gval);
        x->accumulate_grad(CpuTensor::from_f32(std::move(out), x->data.shape()));
    };
    record_named(tag, OpType::Unary, {x}, v);
    return v;
}

inline Value* matmul_qk(Graph& g, Value* q, Value* k, uint32_t bsz, uint32_t seq, uint32_t dim, std::string_view tag = {}) {
    auto out = matmul_batched_qk(q->data, k->data, bsz, seq, dim);
    auto* v = g.node(out, q->requires_grad || k->requires_grad, std::string(tag));
    v->parents = {q, k};
    v->backward_fn = [q, k, v, bsz, seq, dim]() {
        if (!v->grad) return;
        auto d_scores = to_float_vec(*v->grad);
        auto kf = to_float_vec(k->data);
        auto qf = to_float_vec(q->data);
        std::vector<float> dq(qf.size(), 0.0f);
        std::vector<float> dk(kf.size(), 0.0f);
        for (uint32_t b0 = 0; b0 < bsz; ++b0) {
            for (uint32_t i = 0; i < seq; ++i) {
                for (uint32_t j = 0; j < seq; ++j) {
                    float ds = d_scores[(static_cast<size_t>(b0) * seq + i) * seq + j];
                    const size_t q_base = (static_cast<size_t>(b0) * seq + i) * dim;
                    const size_t k_base = (static_cast<size_t>(b0) * seq + j) * dim;
                    for (uint32_t d = 0; d < dim; ++d) {
                        dq[q_base + d] += ds * kf[k_base + d];
                        dk[k_base + d] += ds * qf[q_base + d];
                    }
                }
            }
        }
        if (q->requires_grad) {
            q->accumulate_grad(CpuTensor::from_f32(std::move(dq), {bsz, seq, dim}));
        }
        if (k->requires_grad) {
            k->accumulate_grad(CpuTensor::from_f32(std::move(dk), {bsz, seq, dim}));
        }
    };
    record_named(tag, OpType::Binary, {q, k}, v);
    return v;
}

inline Value* matmul_av(Graph& g, Value* attn, Value* v_in, uint32_t bsz, uint32_t seq, uint32_t dim, std::string_view tag = {}) {
    auto out = matmul_batched_av(attn->data, v_in->data, bsz, seq, dim);
    auto* v = g.node(out, attn->requires_grad || v_in->requires_grad, std::string(tag));
    v->parents = {attn, v_in};
    v->backward_fn = [attn, v_in, v, bsz, seq, dim]() {
        if (!v->grad) return;
        auto d_out = to_float_vec(*v->grad);
        auto af = to_float_vec(attn->data);
        auto vf = to_float_vec(v_in->data);
        std::vector<float> d_attn(af.size(), 0.0f);
        std::vector<float> d_v(vf.size(), 0.0f);
        for (uint32_t b0 = 0; b0 < bsz; ++b0) {
            for (uint32_t i = 0; i < seq; ++i) {
                for (uint32_t j = 0; j < seq; ++j) {
                    float acc = 0.0f;
                    for (uint32_t d = 0; d < dim; ++d) {
                        const size_t out_idx = (static_cast<size_t>(b0) * seq + i) * dim + d;
                        const size_t v_idx = (static_cast<size_t>(b0) * seq + j) * dim + d;
                        acc += d_out[out_idx] * vf[v_idx];
                        d_v[v_idx] += d_out[out_idx] * af[(static_cast<size_t>(b0) * seq + i) * seq + j];
                    }
                    d_attn[(static_cast<size_t>(b0) * seq + i) * seq + j] += acc;
                }
            }
        }
        if (attn->requires_grad) {
            attn->accumulate_grad(CpuTensor::from_f32(std::move(d_attn), {bsz, seq, seq}));
        }
        if (v_in->requires_grad) {
            v_in->accumulate_grad(CpuTensor::from_f32(std::move(d_v), {bsz, seq, dim}));
        }
    };
    record_named(tag, OpType::Binary, {attn, v_in}, v);
    return v;
}

}  // namespace tensordiffgrad::core
