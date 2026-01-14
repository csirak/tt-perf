// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace tensordiffgradv2 {

enum class DType : uint32_t {
    kBF16 = 0,
    kF32 = 1,
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::kBF16: return 2;
        case DType::kF32: return 4;
    }
    throw std::runtime_error("Unknown dtype");
}

inline std::string dtype_name(DType dtype) {
    switch (dtype) {
        case DType::kBF16: return "bf16";
        case DType::kF32: return "f32";
    }
    throw std::runtime_error("Unknown dtype");
}

}  // namespace tensordiffgradv2
