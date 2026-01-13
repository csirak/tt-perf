// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Simple libtorch comparator for tensordiff binary tensors.

#include "tensordiff/core/serialize.hpp"
#include "tensordiff/core/torch_bridge.hpp"

#include <torch/torch.h>
#include <filesystem>
#include <string>
#include <iostream>

using tensordiff::core::CpuTensor;
using tensordiff::core::load_tensor;
using tensordiff::core::to_torch;

int main(int argc, char** argv) {
    std::filesystem::path dir = "tensordiff/outputs/ttnn";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--dir" && i + 1 < argc) {
            dir = argv[i + 1];
            ++i;
        }
    }

    auto cpu_ref = load_tensor((dir / "cpu_ref.bin").string());
    auto ttnn_roundtrip = load_tensor((dir / "ttnn_roundtrip.bin").string());

    auto a = to_torch(cpu_ref, torch::kBFloat16);
    auto b = to_torch(ttnn_roundtrip, torch::kBFloat16);

    auto af = a.to(torch::kFloat32).flatten();
    auto bf = b.to(torch::kFloat32).flatten();
    auto diff = (af - bf).abs();

    auto max_abs = diff.max().item<double>();
    auto mean_abs = diff.mean().item<double>();
    auto rel_l2 = (diff.pow(2).sum().sqrt() / bf.pow(2).sum().sqrt()).item<double>();

    std::cout << "# tensordiff libtorch compare: numel=" << diff.numel() << "\n";
    std::cout << "# max_abs=" << max_abs << " mean_abs=" << mean_abs << " rel_l2=" << rel_l2 << "\n";
    return 0;
}
