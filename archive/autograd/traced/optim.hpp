// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Traced Optimizers
// SGD optimizer for traced layers.

#pragma once

#include "nn.hpp"
#include <vector>
#include <initializer_list>

namespace traced {

struct TracedSGD {
    std::vector<TracedLinear*> layers;
    float lr;

    TracedSGD(std::initializer_list<TracedLinear*> ls, float learning_rate)
        : layers(ls), lr(learning_rate) {}

    void step() {
        for (auto* layer : layers) {
            layer->sgd_step(lr);
        }
    }

    // No zero_grad needed - gradients are overwritten each backward pass
};

}  // namespace traced
