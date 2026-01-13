// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace tensordiff::core {

struct BFloat16 {
    uint16_t bits = 0;

    constexpr BFloat16() = default;

    template <class T>
    requires std::is_arithmetic_v<T>
    explicit BFloat16(T v) noexcept : bits(from_float(static_cast<float>(v))) {}

    static BFloat16 truncate(float val) {
        uint32_t u32 = std::bit_cast<uint32_t>(val);
        return BFloat16::from_bits(static_cast<uint16_t>(u32 >> 16));
    }

    static BFloat16 from_bits(uint16_t raw_bits) {
        BFloat16 out;
        out.bits = raw_bits;
        return out;
    }

    operator float() const {
        uint32_t u32 = static_cast<uint32_t>(bits) << 16;
        return std::bit_cast<float>(u32);
    }

private:
    static uint16_t from_float(float val) {
        if (std::isnan(val)) {
            return UINT16_C(0x7FC0);
        }
        uint32_t u32 = std::bit_cast<uint32_t>(val);
        uint32_t rounding_bias = ((u32 >> 16) & 1U) + UINT32_C(0x7FFF);
        return static_cast<uint16_t>((u32 + rounding_bias) >> 16);
    }
};

static_assert(sizeof(BFloat16) == 2, "BFloat16 must be 2 bytes");

}  // namespace tensordiff::core
