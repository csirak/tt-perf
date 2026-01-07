#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Compare C++ traced autograd outputs against PyTorch reference.
#
# Usage:
#   python3 verify/compare_outputs.py                           # Default (cpp)
#   python3 verify/compare_outputs.py --cpp-dir verify/outputs/cpp_fp32acc  # FP32 acc

import argparse
from pathlib import Path
from tensor_io import load_tensor, compare_tensors, print_comparison

PYTORCH_DIR = Path("verify/outputs/pytorch")
DEFAULT_CPP_DIR = Path("verify/outputs/cpp")

# Files to compare (in order of computation)
COMPARE_FILES = [
    "q_proj.bin",
    "k_proj.bin",
    "v_proj.bin",
    "q.bin",
    "k.bin",
    "v.bin",
    "scores.bin",
    "scores_masked.bin",
    "attn_weights.bin",
    "attn_out.bin",
    "attn_merged.bin",
    "output.bin",
]


def main():
    parser = argparse.ArgumentParser(description="Compare C++ outputs vs PyTorch reference")
    parser.add_argument("--cpp-dir", type=str, default=str(DEFAULT_CPP_DIR),
                        help="C++ output directory (default: verify/outputs/cpp)")
    args = parser.parse_args()

    cpp_dir = Path(args.cpp_dir)

    print("=" * 70)
    print("Comparing C++ Traced Autograd vs PyTorch Reference")
    print("=" * 70)
    print(f"PyTorch dir: {PYTORCH_DIR}")
    print(f"C++ dir:     {cpp_dir}")

    # Check directories exist
    if not PYTORCH_DIR.exists():
        print(f"\nERROR: PyTorch output directory not found: {PYTORCH_DIR}")
        print("Run: python3 verify/gen_pytorch_outputs.py")
        return 1

    if not cpp_dir.exists():
        print(f"\nERROR: C++ output directory not found: {cpp_dir}")
        print("Run: make verify-attention-cpp")
        return 1

    print("\n--- Layer-by-Layer Comparison ---")
    results = []

    for filename in COMPARE_FILES:
        pytorch_path = PYTORCH_DIR / filename
        cpp_path = cpp_dir / filename

        if not pytorch_path.exists():
            print(f"  {filename:30s}: SKIP (PyTorch file missing)")
            continue
        if not cpp_path.exists():
            print(f"  {filename:30s}: SKIP (C++ file missing)")
            continue

        # Load tensors
        pytorch_t = load_tensor(str(pytorch_path))
        cpp_t = load_tensor(str(cpp_path))

        # Check shapes match
        if pytorch_t.shape != cpp_t.shape:
            print(f"  {filename:30s}: SHAPE MISMATCH! PyTorch={list(pytorch_t.shape)}, C++={list(cpp_t.shape)}")
            continue

        # Compare
        name = filename.replace(".bin", "")
        result = compare_tensors(name, cpp_t, pytorch_t)
        results.append(result)
        print_comparison(result)

    if not results:
        print("\nNo files to compare!")
        return 1

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    max_diffs = [r['max_diff'] for r in results]
    mean_diffs = [r['mean_diff'] for r in results]
    rel_errors = [r['rel_error'] for r in results]

    print(f"  Best case (min max_diff):  {min(max_diffs):.6f}")
    print(f"  Worst case (max max_diff): {max(max_diffs):.6f}")
    print(f"  Average mean_diff:         {sum(mean_diffs)/len(mean_diffs):.6f}")
    print(f"  Average rel_error:         {sum(rel_errors)/len(rel_errors):.4f}")

    # Find worst offender
    worst = max(results, key=lambda r: r['max_diff'])
    print(f"\n  Worst offender: {worst['name']} (max_diff={worst['max_diff']:.6f})")

    # Pass/fail thresholds
    THRESH_MAX = 0.05   # max absolute difference
    THRESH_REL = 0.15   # 15% relative error

    all_pass = all(r['max_diff'] < THRESH_MAX and r['rel_error'] < THRESH_REL for r in results)

    if all_pass:
        print(f"\n  ✓ ALL PASS (max_diff < {THRESH_MAX}, rel_error < {THRESH_REL})")
        return 0
    else:
        print(f"\n  ✗ SOME FAIL (threshold: max_diff < {THRESH_MAX}, rel_error < {THRESH_REL})")
        for r in results:
            if r['max_diff'] >= THRESH_MAX or r['rel_error'] >= THRESH_REL:
                print(f"    FAIL: {r['name']} (max={r['max_diff']:.6f}, rel={r['rel_error']:.4f})")
        return 1


if __name__ == "__main__":
    exit(main())
