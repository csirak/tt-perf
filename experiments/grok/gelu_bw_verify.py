#!/usr/bin/env python3
"""Verify GELU backward against PyTorch."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tensor_io import load_tensor, compare_tensors, print_comparison


def load_meta(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="GELU backward verify")
    parser.add_argument("--cpp-dir", type=str, default="experiments/grok/outputs/gelu_bw")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cpp_dir = Path(args.cpp_dir)
    meta = load_meta(cpp_dir / "meta.json")
    approx = meta.get("gelu_approx", "none")

    device = torch.device(args.device)
    dtype = torch.bfloat16

    x = load_tensor(str(cpp_dir / "x.bin")).to(device=device, dtype=dtype)
    target = load_tensor(str(cpp_dir / "target.bin")).to(device=device, dtype=dtype)

    x = x.detach().requires_grad_(True)
    out = F.gelu(x, approximate=approx).to(dtype)
    loss = torch.mean((out - target) ** 2)
    loss.backward()

    cpp_out = load_tensor(str(cpp_dir / "out.bin")).to(device=device, dtype=dtype)
    cpp_dx = load_tensor(str(cpp_dir / "d_x.bin")).to(device=device, dtype=dtype)

    results = [
        compare_tensors("out", out, cpp_out),
        compare_tensors("d_x", x.grad, cpp_dx),
    ]
    for r in results:
        print_comparison(r)

    print(f"loss_torch={loss.item():.6f}")


if __name__ == "__main__":
    main()
