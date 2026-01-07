#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Validate TTNN matmul output matches PyTorch CPU.
# Tests the shapes used in GPT-2 benchmark.

import torch
import ttnn
import time

# Test shapes from GPT-2 benchmark
TEST_SHAPES = [
    # (M, K, N, description)
    (8192, 512, 512, "QKV/Output projection"),
    (8192, 512, 2048, "FFN up"),
    (8192, 2048, 512, "FFN down"),
    (8192, 64, 256, "Attention scores (batched)"),
    (8192, 256, 64, "Attention output (batched)"),
]


def compare_tensors(name, torch_out, ttnn_out, atol=0.5, rtol=0.1):
    """Compare two tensors and report differences.

    Uses a hybrid approach:
    - For large values: relative error (rtol=10%)
    - For small values near zero: absolute error (atol=0.5)

    BF16 has ~3 decimal digits of precision.
    """
    diff = torch.abs(torch_out - ttnn_out)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Compute relative error only where torch_out is significant
    abs_torch = torch.abs(torch_out)

    # Hybrid tolerance: max(atol, rtol * |expected|)
    tolerance = torch.maximum(torch.tensor(atol), rtol * abs_torch)
    within_tol = (diff <= tolerance).float().mean().item() * 100

    # Pearson correlation coefficient (standard correctness metric)
    torch_flat = torch_out.flatten().float()
    ttnn_flat = ttnn_out.flatten().float()
    pcc = torch.corrcoef(torch.stack([torch_flat, ttnn_flat]))[0, 1].item()

    # Pass if PCC > 0.99 (99% correlation) or >99% elements within tolerance
    passed = pcc > 0.99 or within_tol > 99.0
    status = "✓ PASS" if passed else "✗ FAIL"

    print(f"  {name}: max_diff={max_diff:.3f}, mean_diff={mean_diff:.4f}, PCC_corr={pcc:.6f}, within_tol={within_tol:.1f}% [{status}]")
    return passed


def test_matmul_shape(M, K, N, desc):
    """Test a single matmul shape"""
    print(f"\n### {desc}: [{M}, {K}] @ [{K}, {N}]")

    # Create random inputs in BF16
    torch.manual_seed(42)
    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    b_torch = torch.randn(K, N, dtype=torch.bfloat16)

    # PyTorch CPU reference
    c_torch = torch.matmul(a_torch, b_torch)

    # TTNN
    device = ttnn.open_device(device_id=0)
    try:
        a_ttnn = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b_ttnn = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # Warmup
        for _ in range(3):
            c_ttnn = ttnn.matmul(a_ttnn, b_ttnn)
        ttnn.synchronize_device(device)

        # Timed
        start = time.time()
        for _ in range(10):
            c_ttnn = ttnn.matmul(a_ttnn, b_ttnn)
        ttnn.synchronize_device(device)
        end = time.time()

        avg_time_us = (end - start) / 10 * 1e6
        flops = 2 * M * K * N
        tflops = flops / (avg_time_us * 1e-6) / 1e12

        print(f"  TTNN time: {avg_time_us:.1f} µs, {tflops:.2f} TFLOPS")

        # Compare
        c_ttnn_torch = ttnn.to_torch(c_ttnn).to(torch.bfloat16)
        passed = compare_tensors("Matmul output", c_torch, c_ttnn_torch)

    finally:
        ttnn.close_device(device)

    return passed


def test_matmul_transpose_b(M, K, N, desc):
    """Test matmul with transposed B (like Linear layer: x @ W.T)"""
    print(f"\n### {desc} (transpose_b): [{M}, {K}] @ [{N}, {K}].T")

    torch.manual_seed(42)
    a_torch = torch.randn(M, K, dtype=torch.bfloat16)
    # Weight is stored as [out_features, in_features] = [N, K]
    b_torch = torch.randn(N, K, dtype=torch.bfloat16)

    # PyTorch: a @ b.T
    c_torch = torch.matmul(a_torch, b_torch.T)

    # TTNN
    device = ttnn.open_device(device_id=0)
    try:
        a_ttnn = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b_ttnn = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # TTNN matmul with transpose_b=True
        c_ttnn = ttnn.matmul(a_ttnn, b_ttnn, transpose_b=True)
        ttnn.synchronize_device(device)

        # Compare
        c_ttnn_torch = ttnn.to_torch(c_ttnn).to(torch.bfloat16)
        passed = compare_tensors("Matmul (transpose_b)", c_torch, c_ttnn_torch)

    finally:
        ttnn.close_device(device)

    return passed


def main():
    print("=" * 70)
    print("TTNN vs PyTorch Matmul Correctness Test")
    print("=" * 70)

    all_passed = True

    # Test standard matmuls
    for M, K, N, desc in TEST_SHAPES:
        passed = test_matmul_shape(M, K, N, desc)
        all_passed = all_passed and passed

    # Test transpose_b (Linear layer pattern)
    print("\n" + "=" * 70)
    print("Testing transpose_b (Linear layer pattern)")
    print("=" * 70)

    passed = test_matmul_transpose_b(8192, 512, 512, "Linear 512->512")
    all_passed = all_passed and passed

    passed = test_matmul_transpose_b(8192, 512, 2048, "Linear 512->2048")
    all_passed = all_passed and passed

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All tests PASSED")
    else:
        print("✗ Some tests FAILED")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
