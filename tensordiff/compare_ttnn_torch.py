#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parent))
from td_io import load_tensor


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = (af - bf).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    mean_abs = diff.mean().item() if diff.numel() else 0.0
    rel_l2 = (diff.pow(2).sum().sqrt() / bf.pow(2).sum().sqrt()).item() if diff.numel() else 0.0
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel_l2": rel_l2,
        "numel": diff.numel(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="tensordiff/outputs/ttnn", help="output dir from C++ run")
    args = parser.parse_args()

    out_dir = Path(args.dir)
    cpu_ref = load_tensor(out_dir / "cpu_ref.bin")
    ttnn_roundtrip = load_tensor(out_dir / "ttnn_roundtrip.bin")

    result = compare(cpu_ref, ttnn_roundtrip)
    print(f"# tensordiff torch compare: numel={result['numel']}")
    print(f"# max_abs={result['max_abs']:.6f} mean_abs={result['mean_abs']:.6f} rel_l2={result['rel_l2']:.6f}")


if __name__ == "__main__":
    main()
