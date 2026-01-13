#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tensordiff.td_io import load_tensor  # noqa: E402


def compare(a: torch.Tensor, b: torch.Tensor):
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    diff = (a - b).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel_l2 = (diff.flatten().norm() / b.flatten().norm()).item() if b.numel() > 0 else 0.0
    return max_abs, mean_abs, rel_l2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.dir)

    pairs = [
        ("output.bin", "torch_output.bin"),
        ("input_grad.bin", "torch_input_grad.bin"),
        ("wq_grad.bin", "torch_wq_grad.bin"),
        ("wk_grad.bin", "torch_wk_grad.bin"),
        ("wv_grad.bin", "torch_wv_grad.bin"),
        ("wo_grad.bin", "torch_wo_grad.bin"),
        ("w1_grad.bin", "torch_w1_grad.bin"),
        ("w2_grad.bin", "torch_w2_grad.bin"),
        ("bq_grad.bin", "torch_bq_grad.bin"),
        ("bk_grad.bin", "torch_bk_grad.bin"),
        ("bv_grad.bin", "torch_bv_grad.bin"),
        ("bo_grad.bin", "torch_bo_grad.bin"),
        ("b1_grad.bin", "torch_b1_grad.bin"),
        ("b2_grad.bin", "torch_b2_grad.bin"),
        ("ln1_gamma_grad.bin", "torch_ln1_gamma_grad.bin"),
        ("ln1_beta_grad.bin", "torch_ln1_beta_grad.bin"),
        ("ln2_gamma_grad.bin", "torch_ln2_gamma_grad.bin"),
        ("ln2_beta_grad.bin", "torch_ln2_beta_grad.bin"),
    ]

    for a_name, b_name in pairs:
        a_path = out_dir / a_name
        b_path = out_dir / b_name
        if not a_path.exists() or not b_path.exists():
            continue
        a = load_tensor(str(a_path)).to(torch.float32)
        b = load_tensor(str(b_path)).to(torch.float32)
        max_abs, mean_abs, rel_l2 = compare(a, b)
        print(f"{a_name}: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rel_l2={rel_l2:.6g}")


if __name__ == "__main__":
    main()
