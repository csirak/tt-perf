// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Optimizers for TTNN Autograd

#pragma once

#include "autograd.hpp"

namespace autograd {

// Stochastic Gradient Descent
struct SGD {
    std::vector<ValuePtr> params;
    float lr;

    SGD(std::vector<ValuePtr> params_, float lr_)
        : params(std::move(params_)), lr(lr_) {}

    // Update: param.data = param.data - lr * param.grad
    void step() {
        for (auto& p : params) {
            if (!p->grad.has_value()) continue;
            auto update = ttnn::multiply(p->grad.value(), lr);
            p->data = ttnn::subtract(p->data, update);
        }
    }

    // Clear all gradients - just reset to nullopt, accumulate_grad handles first assignment
    void zero_grad() {
        for (auto& p : params) {
            p->grad = std::nullopt;
        }
    }
};

}  // namespace autograd
