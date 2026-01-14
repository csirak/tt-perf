// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// FFN test using tensordiffgradv2 with TtnnTensor backend.
// Demonstrates autograd + op logging on Wormhole device.

// Include paths work for both standalone and tt-metal builds
#include "backends/ttnn_tensor.hpp"
#include "autograd/graph.hpp"

#include <ttnn/device.hpp>

#include <iostream>
#include <string>

using namespace tensordiffgradv2;

int main(int argc, char* argv[]) {
    // Config - tile aligned dimensions
    const int64_t B = 1;
    const int64_t S = 32;
    const int64_t D = 64;
    const int64_t F = 256;

    // Output dir: outputs/<test_name>_<dtype>
    std::string log_dir;
    if (argc > 1) {
        log_dir = argv[1];
    } else {
        log_dir = "outputs/ttnn_ffn_bf16";
    }

    std::cout << "tensordiffgradv2 TTNN FFN test\n";
    std::cout << "  dtype: bf16 (TTNN always BF16)\n";
    std::cout << "  shape: B=" << B << " S=" << S << " D=" << D << " F=" << F << "\n";

    // Open device
    std::cout << "\nOpening device...\n";
    auto device = ttnn::open_mesh_device(0);

    // Create tensors on device with different seeds for variety
    std::cout << "Creating tensors on device...\n";
    auto x = TtnnTensor::randn({B, S, D}, *device, 42);
    auto w1 = TtnnTensor::randn({D, F}, *device, 43);
    auto b1 = TtnnTensor::randn({1, 1, F}, *device, 44);
    auto w2 = TtnnTensor::randn({F, D}, *device, 45);
    auto b2 = TtnnTensor::randn({1, 1, D}, *device, 46);

    // Create graph with logging
    Graph graph;
    std::cout << "  logging to: " << log_dir << "\n";
    graph.enable_logging(log_dir);

    // Wrap as Values
    auto* vx = graph.input(x, "x");
    auto* vw1 = graph.param(w1, "w1");
    auto* vb1 = graph.param(b1, "b1");
    auto* vw2 = graph.param(w2, "w2");
    auto* vb2 = graph.param(b2, "b2");

    // Matmul opts - demonstrate different configs
    MatmulOpts mm1_opts{.math_fidelity = 2, .fp32_acc = true, .approx = false};
    MatmulOpts mm2_opts{.math_fidelity = 2, .fp32_acc = false, .approx = false};  // defaults

    // Forward pass: FFN
    std::cout << "\nRunning forward pass...\n";

    // h1 = x @ w1 (with fp32 accumulation)
    auto* h1 = vx->matmul(vw1, "ffn_mm1", mm1_opts);

    // h2 = h1 + b1 (broadcast)
    auto* h2 = h1->add(vb1, "ffn_bias1");

    // h3 = gelu(h2)
    auto* h3 = h2->gelu("ffn_gelu");

    // h4 = h3 @ w2 (default opts)
    auto* h4 = h3->matmul(vw2, "ffn_mm2", mm2_opts);

    // out = h4 + b2
    auto* out = h4->add(vb2, "ffn_bias2");

    // loss = sum(out)
    auto* loss = out->sum("loss");

    std::cout << "Forward pass complete.\n";

    // Backward pass
    std::cout << "\nRunning backward pass...\n";
    graph.backward(loss);

    std::cout << "Backward pass complete.\n";

    // Print some info
    std::cout << "\nGradients computed:\n";
    if (vx->grad) {
        auto s = vx->grad->shape();
        std::cout << "  x.grad shape: " << s[0] << "x" << s[1] << "x" << s[2] << "\n";
    }
    if (vw1->grad) {
        auto s = vw1->grad->shape();
        std::cout << "  w1.grad shape: " << s[0] << "x" << s[1] << "\n";
    }
    if (vw2->grad) {
        auto s = vw2->grad->shape();
        std::cout << "  w2.grad shape: " << s[0] << "x" << s[1] << "\n";
    }

    std::cout << "\nOp log written to: " << log_dir << "/ops.tsv\n";

    // Close device
    std::cout << "\nClosing device...\n";
    ttnn::close_device(*device);

    std::cout << "Done.\n";
    return 0;
}
