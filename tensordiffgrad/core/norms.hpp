// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ops.hpp"

#include <cmath>
#include <random>
#include <string>
#include <string_view>

namespace tensordiffgrad::core
{

    enum class NormKind
    {
        LayerNorm,
        RMSNorm,
        DyT,
    };

    inline CpuTensor broadcast_1x1d(const CpuTensor &v, uint32_t bsz, uint32_t seq, uint32_t dim)
    {
        auto vf = to_float_vec(v);
        if (vf.size() != dim)
        {
            throw std::runtime_error("broadcast_1x1d: expected dim-sized vector");
        }
        std::vector<float> out(static_cast<size_t>(bsz) * seq * dim);
        for (uint32_t b0 = 0; b0 < bsz; ++b0)
        {
            for (uint32_t s = 0; s < seq; ++s)
            {
                const size_t base = (static_cast<size_t>(b0) * seq + s) * dim;
                for (uint32_t d = 0; d < dim; ++d)
                {
                    out[base + d] = vf[d];
                }
            }
        }
        return CpuTensor::from_f32(std::move(out), {bsz, seq, dim});
    }

    struct NormModule
    {
        NormKind kind;
        uint32_t dim;
        float eps;
        Value *gamma = nullptr;
        Value *beta = nullptr;
        Value *alpha = nullptr; // DyT only (scalar)

        NormModule(NormKind k, uint32_t d, float eps_) : kind(k), dim(d), eps(eps_) {}

        void init(Graph &g, std::string_view name_prefix = {}, float alpha_init = 1.0f)
        {
            const std::string prefix = std::string(name_prefix);
            std::vector<float> ones(dim, 1.0f);
            std::vector<float> zeros(dim, 0.0f);
            gamma = g.leaf(CpuTensor::from_f32(ones, {1, 1, dim}), true, prefix + "gamma");
            beta = g.leaf(CpuTensor::from_f32(zeros, {1, 1, dim}), true, prefix + "beta");
            if (kind == NormKind::DyT)
            {
                CpuTensor a({1}, DType::kF32);
                *reinterpret_cast<float *>(a.raw().data()) = alpha_init;
                alpha = g.leaf(std::move(a), true, prefix + "alpha");
            }
        }

        Value *forward(Graph &g, Value *x, uint32_t bsz, uint32_t seq, std::string_view tag = {})
        {
            if (kind == NormKind::LayerNorm)
            {
                return layernorm(g, x, gamma, beta, bsz, seq, dim, eps, tag);
            }
            if (kind == NormKind::RMSNorm)
            {
                return rmsnorm(g, x, gamma, beta, bsz, seq, dim, eps, tag);
            }
            return dyt(g, x, alpha, gamma, beta, bsz, seq, dim, tag);
        }

        static Value *layernorm(Graph &g,
                                Value *x,
                                Value *gamma,
                                Value *beta,
                                uint32_t bsz,
                                uint32_t seq,
                                uint32_t dim,
                                float eps,
                                std::string_view tag)
        {
            auto xf = to_float_vec(x->data);
            auto gf = to_float_vec(gamma->data);
            auto bf = to_float_vec(beta->data);
            std::vector<float> out(xf.size());
            std::vector<float> xhat(xf.size());
            std::vector<float> inv_std(static_cast<size_t>(bsz) * seq);

            for (uint32_t b0 = 0; b0 < bsz; ++b0)
            {
                for (uint32_t s = 0; s < seq; ++s)
                {
                    const size_t base = (static_cast<size_t>(b0) * seq + s) * dim;
                    float mean = 0.0f;
                    for (uint32_t d = 0; d < dim; ++d)
                        mean += xf[base + d];
                    mean /= static_cast<float>(dim);
                    float var = 0.0f;
                    for (uint32_t d = 0; d < dim; ++d)
                    {
                        float v = xf[base + d] - mean;
                        var += v * v;
                    }
                    var /= static_cast<float>(dim);
                    float inv = 1.0f / std::sqrt(var + eps);
                    inv_std[static_cast<size_t>(b0) * seq + s] = inv;
                    for (uint32_t d = 0; d < dim; ++d)
                    {
                        float xn = (xf[base + d] - mean) * inv;
                        xhat[base + d] = xn;
                        out[base + d] = xn * gf[d] + bf[d];
                    }
                }
            }

            CpuTensor out_t = CpuTensor::from_f32(std::move(out), {bsz, seq, dim});
            auto *v = g.node(out_t, x->requires_grad || gamma->requires_grad || beta->requires_grad, std::string(tag));
            v->parents = {x, gamma, beta};

            v->backward_fn = [x, gamma, beta, v, bsz, seq, dim, xhat, inv_std]()
            {
                if (!v->grad)
                    return;
                auto dout = to_float_vec(*v->grad);
                auto gf = to_float_vec(gamma->data);
                std::vector<float> dgamma(dim, 0.0f);
                std::vector<float> dbeta(dim, 0.0f);
                std::vector<float> dx(dout.size(), 0.0f);

                for (uint32_t b0 = 0; b0 < bsz; ++b0)
                {
                    for (uint32_t s = 0; s < seq; ++s)
                    {
                        const size_t base = (static_cast<size_t>(b0) * seq + s) * dim;
                        float sum_dxhat = 0.0f;
                        float sum_dxhat_xhat = 0.0f;
                        for (uint32_t d = 0; d < dim; ++d)
                        {
                            float dxhat = dout[base + d] * gf[d];
                            sum_dxhat += dxhat;
                            sum_dxhat_xhat += dxhat * xhat[base + d];
                            dgamma[d] += dout[base + d] * xhat[base + d];
                            dbeta[d] += dout[base + d];
                        }
                        float inv = inv_std[static_cast<size_t>(b0) * seq + s];
                        for (uint32_t d = 0; d < dim; ++d)
                        {
                            float dxhat = dout[base + d] * gf[d];
                            float term1 = dxhat * static_cast<float>(dim);
                            float term2 = sum_dxhat;
                            float term3 = xhat[base + d] * sum_dxhat_xhat;
                            dx[base + d] = (inv / static_cast<float>(dim)) * (term1 - term2 - term3);
                        }
                    }
                }

                if (x->requires_grad)
                {
                    x->accumulate_grad(CpuTensor::from_f32(std::move(dx), {bsz, seq, dim}));
                }
                if (gamma->requires_grad)
                {
                    gamma->accumulate_grad(CpuTensor::from_f32(std::move(dgamma), {1, 1, dim}));
                }
                if (beta->requires_grad)
                {
                    beta->accumulate_grad(CpuTensor::from_f32(std::move(dbeta), {1, 1, dim}));
                }
            };

            record_named(tag, OpType::Unary, {x, gamma, beta}, v, std::string("eps=") + std::to_string(eps));
            return v;
        }

        static Value *rmsnorm(Graph &g,
                              Value *x,
                              Value *gamma,
                              Value *beta,
                              uint32_t bsz,
                              uint32_t seq,
                              uint32_t dim,
                              float eps,
                              std::string_view tag)
        {
            auto xf = to_float_vec(x->data);
            auto gf = to_float_vec(gamma->data);
            auto bf = to_float_vec(beta->data);
            std::vector<float> out(xf.size());
            std::vector<float> xnorm(xf.size());
            std::vector<float> rstd(static_cast<size_t>(bsz) * seq);

            for (uint32_t b0 = 0; b0 < bsz; ++b0)
            {
                for (uint32_t s = 0; s < seq; ++s)
                {
                    const size_t base = (static_cast<size_t>(b0) * seq + s) * dim;
                    float mean_sq = 0.0f;
                    for (uint32_t d = 0; d < dim; ++d)
                    {
                        float v = xf[base + d];
                        mean_sq += v * v;
                    }
                    mean_sq /= static_cast<float>(dim);
                    float inv = 1.0f / std::sqrt(mean_sq + eps);
                    rstd[static_cast<size_t>(b0) * seq + s] = inv;
                    for (uint32_t d = 0; d < dim; ++d)
                    {
                        float xn = xf[base + d] * inv;
                        xnorm[base + d] = xn;
                        out[base + d] = xn * gf[d] + bf[d];
                    }
                }
            }

            CpuTensor out_t = CpuTensor::from_f32(std::move(out), {bsz, seq, dim});
            auto *v = g.node(out_t, x->requires_grad || gamma->requires_grad || beta->requires_grad, std::string(tag));
            v->parents = {x, gamma, beta};

            v->backward_fn = [x, gamma, beta, v, bsz, seq, dim, xnorm, rstd]()
            {
                if (!v->grad)
                    return;
                auto dout = to_float_vec(*v->grad);
                std::vector<float> dgamma(dim, 0.0f);
                std::vector<float> dbeta(dim, 0.0f);
                std::vector<float> dx(dout.size(), 0.0f);
                auto gf = to_float_vec(gamma->data);
                auto xf = to_float_vec(x->data);

                for (uint32_t b0 = 0; b0 < bsz; ++b0)
                {
                    for (uint32_t s = 0; s < seq; ++s)
                    {
                        const size_t base = (static_cast<size_t>(b0) * seq + s) * dim;
                        float mean_gx = 0.0f;
                        for (uint32_t d = 0; d < dim; ++d)
                        {
                            float dxn = dout[base + d] * gf[d];
                            mean_gx += dxn * xf[base + d];
                            dgamma[d] += dout[base + d] * xnorm[base + d];
                            dbeta[d] += dout[base + d];
                        }
                        mean_gx /= static_cast<float>(dim);
                        float inv = rstd[static_cast<size_t>(b0) * seq + s];
                        float inv3 = inv * inv * inv;
                        for (uint32_t d = 0; d < dim; ++d)
                        {
                            float dxn = dout[base + d] * gf[d];
                            float term1 = dxn * inv;
                            float term2 = xf[base + d] * inv3 * mean_gx;
                            dx[base + d] = term1 - term2;
                        }
                    }
                }

                if (x->requires_grad)
                {
                    x->accumulate_grad(CpuTensor::from_f32(std::move(dx), {bsz, seq, dim}));
                }
                if (gamma->requires_grad)
                {
                    gamma->accumulate_grad(CpuTensor::from_f32(std::move(dgamma), {1, 1, dim}));
                }
                if (beta->requires_grad)
                {
                    beta->accumulate_grad(CpuTensor::from_f32(std::move(dbeta), {1, 1, dim}));
                }
            };

            record_named(tag, OpType::Unary, {x, gamma, beta}, v, std::string("eps=") + std::to_string(eps));
            return v;
        }

        static Value *dyt(Graph &g,
                          Value *x,
                          Value *alpha,
                          Value *gamma,
                          Value *beta,
                          uint32_t bsz,
                          uint32_t seq,
                          uint32_t dim,
                          std::string_view tag)
        {
            auto xf = to_float_vec(x->data);
            auto gf = to_float_vec(gamma->data);
            auto bf = to_float_vec(beta->data);
            float a = to_float_vec(alpha->data)[0];
            std::vector<float> tanh_out(xf.size());
            std::vector<float> out(xf.size());
            for (size_t i = 0; i < xf.size(); ++i)
            {
                float t = std::tanh(a * xf[i]);
                tanh_out[i] = t;
                out[i] = gf[i % dim] * t + bf[i % dim];
            }
            CpuTensor out_t = CpuTensor::from_f32(std::move(out), {bsz, seq, dim});
            auto *v = g.node(out_t, x->requires_grad || gamma->requires_grad || beta->requires_grad || alpha->requires_grad, std::string(tag));
            v->parents = {x, alpha, gamma, beta};

            v->backward_fn = [x, alpha, gamma, beta, v, tanh_out, bsz, seq, dim]()
            {
                if (!v->grad)
                    return;
                auto dout = to_float_vec(*v->grad);
                auto gf = to_float_vec(gamma->data);
                float a = to_float_vec(alpha->data)[0];
                std::vector<float> dgamma(dim, 0.0f);
                std::vector<float> dbeta(dim, 0.0f);
                std::vector<float> dx(dout.size(), 0.0f);
                float d_alpha = 0.0f;

                for (size_t i = 0; i < dout.size(); ++i)
                {
                    const uint32_t d = static_cast<uint32_t>(i % dim);
                    dbeta[d] += dout[i];
                    dgamma[d] += dout[i] * tanh_out[i];
                    float one_minus = 1.0f - tanh_out[i] * tanh_out[i];
                    float d_x_scaled = dout[i] * gf[d] * one_minus;
                    dx[i] = d_x_scaled * a;
                    d_alpha += d_x_scaled * to_float_vec(x->data)[i];
                }

                if (x->requires_grad)
                {
                    x->accumulate_grad(CpuTensor::from_f32(std::move(dx), {bsz, seq, dim}));
                }
                if (gamma->requires_grad)
                {
                    gamma->accumulate_grad(CpuTensor::from_f32(std::move(dgamma), {1, 1, dim}));
                }
                if (beta->requires_grad)
                {
                    beta->accumulate_grad(CpuTensor::from_f32(std::move(dbeta), {1, 1, dim}));
                }
                if (alpha->requires_grad)
                {
                    CpuTensor da({1}, DType::kF32);
                    *reinterpret_cast<float *>(da.raw().data()) = d_alpha;
                    alpha->accumulate_grad(da);
                }
            };

            record_named(tag, OpType::Unary, {x, alpha, gamma, beta}, v);
            return v;
        }
    };

} // namespace tensordiffgrad::core
