// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "dtype.hpp"

#include "bfloat16.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace tensordiff::core {

class CpuTensor {
public:
    CpuTensor(std::vector<uint32_t> shape, DType dtype)
        : shape_(std::move(shape)), dtype_(dtype), data_(numel() * dtype_size(dtype)) {}

    static CpuTensor from_bf16(std::vector<BFloat16> values, std::vector<uint32_t> shape) {
        CpuTensor t(std::move(shape), DType::kBF16);
        auto* dst = reinterpret_cast<BFloat16*>(t.data_.data());
        if (values.size() != t.numel()) {
            throw std::runtime_error("from_bf16: value count mismatch");
        }
        std::memcpy(dst, values.data(), values.size() * sizeof(BFloat16));
        return t;
    }

    static CpuTensor from_f32(std::vector<float> values, std::vector<uint32_t> shape) {
        CpuTensor t(std::move(shape), DType::kF32);
        auto* dst = reinterpret_cast<float*>(t.data_.data());
        if (values.size() != t.numel()) {
            throw std::runtime_error("from_f32: value count mismatch");
        }
        std::memcpy(dst, values.data(), values.size() * sizeof(float));
        return t;
    }

    size_t numel() const {
        return std::accumulate(shape_.begin(), shape_.end(), size_t{1}, std::multiplies<>());
    }

    const std::vector<uint32_t>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }

    void fill_ramp(float start, float step) {
        if (dtype_ == DType::kBF16) {
            auto* dst = reinterpret_cast<BFloat16*>(data_.data());
            float v = start;
            for (size_t i = 0; i < numel(); ++i, v += step) {
                dst[i] = BFloat16(v);
            }
            return;
        }
        if (dtype_ == DType::kF32) {
            auto* dst = reinterpret_cast<float*>(data_.data());
            float v = start;
            for (size_t i = 0; i < numel(); ++i, v += step) {
                dst[i] = v;
            }
            return;
        }
        throw std::runtime_error("fill_ramp: unsupported dtype");
    }

    std::vector<BFloat16> to_bf16_vector() const {
        std::vector<BFloat16> out(numel());
        if (dtype_ == DType::kBF16) {
            std::memcpy(out.data(), data_.data(), out.size() * sizeof(BFloat16));
            return out;
        }
        if (dtype_ == DType::kF32) {
            const auto* src = reinterpret_cast<const float*>(data_.data());
            for (size_t i = 0; i < out.size(); ++i) {
                out[i] = BFloat16(src[i]);
            }
            return out;
        }
        throw std::runtime_error("to_bf16_vector: unsupported dtype");
    }

    const std::vector<uint8_t>& raw() const { return data_; }
    std::vector<uint8_t>& raw() { return data_; }

private:
    std::vector<uint32_t> shape_;
    DType dtype_;
    std::vector<uint8_t> data_;
};

}  // namespace tensordiff::core
