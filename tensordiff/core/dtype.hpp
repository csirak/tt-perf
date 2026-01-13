// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <stdexcept>

namespace tensordiff::core {

enum class DType : uint8_t {
    kBF16 = 0,
    kF32 = 1,
    kI32 = 2,
    kU32 = 3,
};

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::kBF16: return "bf16";
        case DType::kF32: return "f32";
        case DType::kI32: return "i32";
        case DType::kU32: return "u32";
        default: return "unknown";
    }
}

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::kBF16: return 2;
        case DType::kF32: return 4;
        case DType::kI32: return 4;
        case DType::kU32: return 4;
        default: throw std::runtime_error("Unknown dtype size");
    }
}

inline DType dtype_from_string(std::string_view s) {
    if (s == "bf16") return DType::kBF16;
    if (s == "f32") return DType::kF32;
    if (s == "i32") return DType::kI32;
    if (s == "u32") return DType::kU32;
    throw std::runtime_error("Unsupported dtype string: " + std::string(s));
}

// Global dtype registry (process-wide).
class DTypeRegistry {
public:
    static void set_default(DType dt) {
        default_ref() = dt;
    }

    static DType get_default() {
        return default_ref();
    }

private:
    static DType& default_ref() {
        static DType dt = DType::kBF16;
        return dt;
    }
};

}  // namespace tensordiff::core
