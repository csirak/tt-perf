// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MLP Training Experiment: 2-layer MLP with MSE loss
// Compares performance with PyTorch implementation

#include "dynamic/autograd.hpp"
#include "dynamic/ops.hpp"
#include "dynamic/nn.hpp"
#include "dynamic/optim.hpp"
#include "common.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>

#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#endif

using namespace autograd;

// MSE Loss: mean((pred - target)^2)
inline ValuePtr mse_loss(ValuePtr pred, ValuePtr target) {
    auto diff = sub(pred, target);
    auto sq = mul(diff, diff);
    return reduce_mean(sq);
}

// 2-layer MLP: input -> Linear -> ReLU -> Linear -> output
struct MLP {
    Linear fc1;
    Linear fc2;

    MLP(uint32_t in_features, uint32_t hidden, uint32_t out_features, MeshDevice& device)
        : fc1(in_features, hidden, device),
          fc2(hidden, out_features, device) {}

    ValuePtr forward(ValuePtr x) {
        auto h = autograd::relu(fc1.forward(x));
        return fc2.forward(h);
    }

    std::vector<ValuePtr> parameters() {
        return {fc1.weight, fc1.bias, fc2.weight, fc2.bias};
    }
};

int main() {
    test::DeviceGuard dg;
    auto& device = dg.get();

    // Config
    const uint32_t batch_size = 1024;
    const uint32_t in_features = 512;
    const uint32_t hidden = 256;
    const uint32_t out_features = 128;
    const float lr = 0.1f;  // Higher LR to compensate for mean reduction (1/numel) scaling
    const int num_iters = 10;
    const int warmup_iters = 2;

    std::cout << "MLP Training Experiment (TTNN C++)\n";
    std::cout << "==================================\n";
    std::cout << "Batch size: " << batch_size << "\n";
    std::cout << "Architecture: " << in_features << " -> " << hidden << " -> " << out_features << "\n";
    std::cout << "Learning rate: " << lr << "\n";
    std::cout << "Iterations: " << num_iters << " (warmup: " << warmup_iters << ")\n\n";

    // Create model and optimizer
    MLP model(in_features, hidden, out_features, device);
    SGD sgd(model.parameters(), lr);

    // Create fixed random input and target
    auto x = make_random(ttnn::Shape({batch_size, in_features}), device, -1.0f, 1.0f, true, "input");
    auto target = make_random(ttnn::Shape({batch_size, out_features}), device, -1.0f, 1.0f, false, "target");

    // Training loop
    std::vector<double> iter_times;
    std::vector<float> losses;

    for (int i = 0; i < num_iters; ++i) {
#ifdef TRACY_ENABLE
        ZoneScopedN("iteration");
#endif
        auto start = std::chrono::high_resolution_clock::now();

        // Forward
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("zero_grad");
#endif
            sgd.zero_grad();
        }
        ValuePtr pred;
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("forward");
#endif
            pred = model.forward(x);
        }
        ValuePtr loss;
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("mse_loss");
#endif
            loss = mse_loss(pred, target);
        }

        // Backward
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("backward");
#endif
            backward(loss, device);
        }

        // Optimizer step
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("sgd_step");
#endif
            sgd.step();
        }

        // Sync and measure time
        {
#ifdef TRACY_ENABLE
            ZoneScopedN("sync");
#endif
            Finish(device.mesh_command_queue());
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Get loss value
        auto loss_val = loss->data.cpu().to_vector<bfloat16>()[0];
        float loss_f = static_cast<float>(loss_val);

        if (i >= warmup_iters) {
            iter_times.push_back(ms);
        }
        losses.push_back(loss_f);

        if (i < 5 || i >= num_iters - 5 || i % 20 == 0) {
            std::cout << "Iter " << std::setw(3) << i
                      << " | Loss: " << std::fixed << std::setprecision(6) << loss_f
                      << " | Time: " << std::setprecision(2) << ms << " ms\n";
        }
    }

    // Statistics
    double total_time = 0;
    for (auto t : iter_times) total_time += t;
    double avg_time = total_time / iter_times.size();

    std::cout << "\n==================================\n";
    std::cout << "Results:\n";
    std::cout << "  Initial loss: " << std::fixed << std::setprecision(6) << losses.front() << "\n";
    std::cout << "  Final loss:   " << losses.back() << "\n";
    std::cout << "  Loss reduced: " << (losses.front() > losses.back() ? "YES" : "NO") << "\n";
    std::cout << "  Avg time/iter (after warmup): " << std::setprecision(2) << avg_time << " ms\n";
    std::cout << "  Throughput: " << std::setprecision(1) << (1000.0 / avg_time) << " iters/sec\n";

    return 0;
}
