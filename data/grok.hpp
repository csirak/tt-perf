// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>
#include <random>

namespace grokking {

struct Batch {
    std::vector<uint32_t> tokens;   // [B, seq_len]
    std::vector<uint32_t> targets;  // [B]
};

struct ModularDivisionDataset {
    uint32_t p;
    float train_frac;
    uint32_t pad_to;
    uint32_t seq_len;
    uint32_t vocab_size;
    uint32_t base_vocab_size;
    uint32_t div_token;
    uint32_t eq_token;
    uint32_t answer_pos;

    std::vector<uint32_t> x_all;
    std::vector<uint32_t> y_all;
    std::vector<uint32_t> z_all;
    std::vector<uint32_t> train_idx;
    std::vector<uint32_t> val_idx;

    std::mt19937 rng;

    ModularDivisionDataset(uint32_t p_, float train_frac_, uint32_t pad_to_, uint32_t seed);

    Batch sample_batch(uint32_t batch_size, bool train_split);

    uint32_t train_size() const { return static_cast<uint32_t>(train_idx.size()); }
    uint32_t val_size() const { return static_cast<uint32_t>(val_idx.size()); }
};

}  // namespace grokking
