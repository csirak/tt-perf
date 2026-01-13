// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cpu_tensor.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace tensordiff::core {

struct CompareResult {
    std::string name;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rel_l2 = 0.0;
    size_t numel = 0;
};

struct Comparison {
    std::string name;
    std::vector<uint32_t> shape;
    std::vector<float> diff;
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double rel_l2 = 0.0;
    size_t numel = 0;
};

using Comparision = Comparison;

inline Comparison compare_tensors_full(const std::string& name, const CpuTensor& a, const CpuTensor& b);

inline std::vector<float> to_float_vec(const CpuTensor& t) {
    std::vector<float> out(t.numel());
    if (t.dtype() == DType::kBF16) {
        const auto* src = reinterpret_cast<const BFloat16*>(t.raw().data());
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = static_cast<float>(src[i]);
        }
        return out;
    }
    if (t.dtype() == DType::kF32) {
        const auto* src = reinterpret_cast<const float*>(t.raw().data());
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = src[i];
        }
        return out;
    }
    throw std::runtime_error("to_float_vec: unsupported dtype");
}

inline CompareResult compare_tensors(const std::string& name, const CpuTensor& a, const CpuTensor& b) {
    auto full = compare_tensors_full(name, a, b);
    CompareResult r;
    r.name = full.name;
    r.max_abs = full.max_abs;
    r.mean_abs = full.mean_abs;
    r.rel_l2 = full.rel_l2;
    r.numel = full.numel;
    return r;
}

inline Comparison compare_tensors_full(const std::string& name, const CpuTensor& a, const CpuTensor& b) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("compare_tensors: shape mismatch");
    }

    auto af = to_float_vec(a);
    auto bf = to_float_vec(b);

    double max_abs = 0.0;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    double sum_sq_ref = 0.0;
    std::vector<float> diff_vec(af.size());

    for (size_t i = 0; i < af.size(); ++i) {
        double diff = static_cast<double>(af[i]) - static_cast<double>(bf[i]);
        double adiff = std::abs(diff);
        max_abs = std::max(max_abs, adiff);
        sum_abs += adiff;
        sum_sq += diff * diff;
        sum_sq_ref += static_cast<double>(bf[i]) * static_cast<double>(bf[i]);
        diff_vec[i] = static_cast<float>(adiff);
    }

    Comparison r;
    r.name = name;
    r.shape = a.shape();
    r.diff = std::move(diff_vec);
    r.max_abs = max_abs;
    r.mean_abs = af.empty() ? 0.0 : (sum_abs / static_cast<double>(af.size()));
    r.rel_l2 = (sum_sq_ref > 1e-12) ? std::sqrt(sum_sq) / std::sqrt(sum_sq_ref) : 0.0;
    r.numel = af.size();
    return r;
}

}  // namespace tensordiff::core
