// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// FFN test using tensordiffgradv2 with TorchTensor backend.
// Demonstrates autograd + op logging.

// Include using relative paths (CMake adds parent as include dir)
#include "tensordiffgradv2/backends/torch_tensor.hpp"
#include "tensordiffgradv2/autograd/graph.hpp"

#include <iostream>
#include <string>

using namespace tensordiffgradv2;

int main(int argc, char* argv[]) {
    // Config
    const int64_t B = 1;
    const int64_t S = 32;
    const int64_t D = 64;
    const int64_t F = 256;

    // Use BF16 to match TTNN
    const DType dtype = DType::kBF16;

    // Output dir: outputs/<test_name>_<dtype>
    // e.g., outputs/torch_ffn_bf16
    std::string log_dir;
    if (argc > 1) {
        log_dir = argv[1];
    } else {
        log_dir = "outputs/torch_ffn_" + dtype_name(dtype);
    }

    std::cout << "tensordiffgradv2 FFN test\n";
    std::cout << "  dtype: " << dtype_name(dtype) << "\n";
    std::cout << "  shape: B=" << B << " S=" << S << " D=" << D << " F=" << F << "\n";

    // Set seed for reproducibility
    torch::manual_seed(42);

    // Create tensors
    auto x = TorchTensor::randn({B, S, D}, dtype);
    auto w1 = TorchTensor::randn({D, F}, dtype);
    auto b1 = TorchTensor::randn({1, 1, F}, dtype);
    auto w2 = TorchTensor::randn({F, D}, dtype);
    auto b2 = TorchTensor::randn({1, 1, D}, dtype);

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

    // Forward pass: FFN
    // h1 = x @ w1
    auto* h1 = vx->matmul(vw1, "ffn_mm1");

    // h2 = h1 + b1 (broadcast)
    // Note: proper broadcast add would need reshape, simplified here
    auto* h2 = h1->add(vb1, "ffn_bias1");

    // h3 = gelu(h2)
    auto* h3 = h2->gelu("ffn_gelu");

    // h4 = h3 @ w2
    auto* h4 = h3->matmul(vw2, "ffn_mm2");

    // out = h4 + b2
    auto* out = h4->add(vb2, "ffn_bias2");

    // loss = sum(out)
    auto* loss = out->sum("loss");

    std::cout << "\nForward pass complete.\n";

    // Backward pass
    graph.backward(loss);

    std::cout << "Backward pass complete.\n";

    // Print some info
    std::cout << "\nGradients computed:\n";
    if (vx->grad) std::cout << "  x.grad shape: " << vx->grad->shape()[0] << "x" << vx->grad->shape()[1] << "x" << vx->grad->shape()[2] << "\n";
    if (vw1->grad) std::cout << "  w1.grad shape: " << vw1->grad->shape()[0] << "x" << vw1->grad->shape()[1] << "\n";
    if (vw2->grad) std::cout << "  w2.grad shape: " << vw2->grad->shape()[0] << "x" << vw2->grad->shape()[1] << "\n";

    std::cout << "\nOp log written to: " << log_dir << "/ops.tsv\n";
    std::cout << "Done.\n";
    return 0;
}
