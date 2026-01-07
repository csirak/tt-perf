// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "grok.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace grokking {

namespace {

uint32_t next_multiple(uint32_t value, uint32_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

uint32_t mod_div(uint32_t x, uint32_t y, uint32_t p) {
    uint64_t base = y;
    uint64_t exp = p - 2;
    uint64_t mod = p;
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        exp >>= 1;
    }
    return static_cast<uint32_t>((result * x) % mod);
}

}  // namespace

ModularDivisionDataset::ModularDivisionDataset(uint32_t p_, float train_frac_, uint32_t pad_to_, uint32_t seed)
    : p(p_),
      train_frac(train_frac_),
      pad_to(pad_to_),
      seq_len(pad_to_),
      vocab_size(0),
      base_vocab_size(0),
      div_token(0),
      eq_token(0),
      answer_pos(3),
      rng(seed) {
    if (p <= 2) {
        throw std::invalid_argument("p must be > 2");
    }
    if (!(train_frac > 0.0f && train_frac < 1.0f)) {
        throw std::invalid_argument("train_frac must be in (0, 1)");
    }
    if (pad_to < 4 || (pad_to % 32) != 0) {
        throw std::invalid_argument("pad_to must be >= 4 and a multiple of 32");
    }

    div_token = p;
    eq_token = p + 1;
    base_vocab_size = p + 2;
    vocab_size = next_multiple(base_vocab_size, 32);

    x_all.reserve(p * (p - 1));
    y_all.reserve(p * (p - 1));
    z_all.reserve(p * (p - 1));
    for (uint32_t x = 0; x < p; ++x) {
        for (uint32_t y = 1; y < p; ++y) {
            x_all.push_back(x);
            y_all.push_back(y);
            z_all.push_back(mod_div(x, y, p));
        }
    }

    const uint32_t total = static_cast<uint32_t>(x_all.size());
    std::vector<uint32_t> perm(total);
    for (uint32_t i = 0; i < total; ++i) {
        perm[i] = i;
    }
    std::shuffle(perm.begin(), perm.end(), rng);
    const uint32_t split = static_cast<uint32_t>(std::floor(static_cast<double>(total) * train_frac));

    train_idx.assign(perm.begin(), perm.begin() + split);
    val_idx.assign(perm.begin() + split, perm.end());
}

Batch ModularDivisionDataset::sample_batch(uint32_t batch_size, bool train_split) {
    if ((batch_size % 32) != 0) {
        throw std::invalid_argument("batch_size must be a multiple of 32");
    }

    const auto& idx_pool = train_split ? train_idx : val_idx;
    if (idx_pool.empty()) {
        throw std::runtime_error(train_split ? "train split is empty" : "val split is empty");
    }

    std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(idx_pool.size() - 1));

    Batch batch;
    batch.tokens.assign(batch_size * seq_len, 0);
    batch.targets.resize(batch_size);

    for (uint32_t i = 0; i < batch_size; ++i) {
        uint32_t idx = idx_pool[dist(rng)];
        uint32_t x = x_all[idx];
        uint32_t y = y_all[idx];
        uint32_t z = z_all[idx];

        uint32_t base = i * seq_len;
        batch.tokens[base + 0] = x;
        batch.tokens[base + 1] = div_token;
        batch.tokens[base + 2] = y;
        batch.tokens[base + 3] = eq_token;
        batch.targets[i] = z;
    }

    return batch;
}

}  // namespace grokking
