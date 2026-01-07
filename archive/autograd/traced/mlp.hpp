// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Template-based N-layer MLP for benchmarking
// Layer count is a compile-time parameter for trace compatibility.

#pragma once

#include "nn.hpp"
#include "ops.hpp"
#include <vector>

namespace traced {

// TracedMLP<N>: N-layer MLP with ReLU activations between layers
// All layers use the same dimension (square weight matrices) for simplicity.
// No final activation - output is raw logits for MSE loss.
template<size_t N>
struct TracedMLP {
    static_assert(N >= 1, "MLP must have at least 1 layer");

    // N linear layers: all dim -> dim (using vector since TracedLinear has no default ctor)
    std::vector<TracedLinear> layers;

    // Forward activation buffers
    std::vector<Tensor> h;            // h[i] = output of layer i (before relu for i < N-1)
    std::vector<Tensor> h_relu;       // h_relu[i] = relu(h[i]) for i < N-1
    std::vector<Tensor> relu_masks;   // masks for backward

    // Backward gradient buffers
    std::vector<Tensor> d_h;          // gradients at each layer output

    // Loss computation buffers (in declaration order)
    Tensor diff;
    Tensor sq;
    Tensor loss;

    // Config (in declaration order)
    float loss_scale;
    float lr;
    uint32_t batch;
    uint32_t dim;

    TracedMLP(uint32_t b, uint32_t d, float learning_rate, MeshDevice& dev)
        : diff(make_zeros(ttnn::Shape({b, d}), dev)),
          sq(make_zeros(ttnn::Shape({b, d}), dev)),
          loss(make_zeros(ttnn::Shape({1, 1}), dev)),
          loss_scale(1.0f / (b * d)),
          lr(learning_rate),
          batch(b),
          dim(d)
    {
        // Reserve space
        layers.reserve(N);
        h.reserve(N);
        d_h.reserve(N);
        h_relu.reserve(N - 1);
        relu_masks.reserve(N - 1);

        // Initialize layers and buffers
        for (size_t i = 0; i < N; ++i) {
            layers.emplace_back(d, d, 0.01f, dev);
            h.push_back(make_zeros(ttnn::Shape({b, d}), dev));
            d_h.push_back(make_zeros(ttnn::Shape({b, d}), dev));
        }
        for (size_t i = 0; i < N - 1; ++i) {
            h_relu.push_back(make_zeros(ttnn::Shape({b, d}), dev));
            relu_masks.push_back(make_zeros(ttnn::Shape({b, d}), dev));
        }
    }

    void forward(const Tensor& x) {
        // First layer
        layers[0].forward(x, h[0]);

        // Middle layers with ReLU
        for (size_t i = 0; i < N - 1; ++i) {
            relu(h[i], h_relu[i], relu_masks[i]);
            layers[i + 1].forward(h_relu[i], h[i + 1]);
        }
        // h[N-1] is the final output (no ReLU)
    }

    void compute_loss(const Tensor& target) {
        subtract(h[N - 1], target, diff);
        multiply(diff, diff, sq);
        mean(sq, loss);
    }

    void backward(const Tensor& x) {
        // MSE gradient: d_pred = 2 * diff * loss_scale
        mse_backward(diff, loss_scale, d_h[N - 1]);

        // Backprop through layers in reverse
        for (size_t i = N - 1; i > 0; --i) {
            // Layer i backward: uses h_relu[i-1] as input
            layers[i].backward(h_relu[i - 1], d_h[i], d_h[i - 1]);
            // ReLU backward
            relu_backward(d_h[i - 1], relu_masks[i - 1], d_h[i - 1]);
        }

        // First layer backward (no d_input needed)
        layers[0].backward_no_input(x, d_h[0]);
    }

    void sgd_step() {
        for (size_t i = 0; i < N; ++i) {
            layers[i].sgd_step(lr);
        }
    }

    void train_step(const Tensor& x, const Tensor& target) {
        forward(x);
        compute_loss(target);
        backward(x);
        sgd_step();
    }

    float get_loss(MeshDevice* dev) {
        tt::tt_metal::distributed::Synchronize(dev, std::nullopt);
        return static_cast<float>(loss.cpu().to_vector<bfloat16>()[0]);
    }
};

}  // namespace traced
