#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent))
from td_io import load_tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="path to first bf16 tensor bin")
    parser.add_argument("--b", required=True, help="path to second bf16 tensor bin")
    parser.add_argument("--out", required=True, help="output png path")
    parser.add_argument("--title", default="diff heatmap", help="plot title")
    parser.add_argument("--vmin", type=float, default=None, help="min color scale")
    parser.add_argument("--vmax", type=float, default=None, help="max color scale")
    args = parser.parse_args()

    a = load_tensor(Path(args.a)).to(dtype=torch.float32).numpy()
    b = load_tensor(Path(args.b)).to(dtype=torch.float32).numpy()

    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch: {a.shape} vs {b.shape}")

    if a.ndim != 2:
        raise SystemExit(f"heatmap expects 2D tensor, got shape {a.shape}")

    diff = np.abs(a - b)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit("matplotlib is required for heatmap rendering") from exc

    plt.figure(figsize=(6, 5))
    im = plt.imshow(diff, cmap="magma", vmin=args.vmin, vmax=args.vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(args.title)
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
