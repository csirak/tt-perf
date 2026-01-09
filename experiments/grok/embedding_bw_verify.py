#!/usr/bin/env python3
"""Verify embedding backward against PyTorch."""

import argparse
import json
from pathlib import Path

import torch

from tensor_io import load_tensor, compare_tensors, print_comparison


def load_meta(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_u32(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        ndim = int.from_bytes(f.read(4), "little")
        shape = [int.from_bytes(f.read(4), "little") for _ in range(ndim)]
        numel = 1
        for d in shape:
            numel *= d
        data = f.read(numel * 4)
    return torch.frombuffer(data, dtype=torch.uint32).clone().reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding backward verify")
    parser.add_argument("--cpp-dir", type=str, default="experiments/grok/outputs/embedding_bw")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cpp_dir = Path(args.cpp_dir)
    meta = load_meta(cpp_dir / "meta.json")

    batch = int(meta["batch"])
    seq = int(meta["seq"])
    vocab = int(meta["vocab"])
    dim = int(meta["dim"])

    device = torch.device(args.device)
    dtype = torch.bfloat16

    indices = load_u32(cpp_dir / "indices_u32.bin").to(device=device, dtype=torch.long)
    target = load_tensor(str(cpp_dir / "target.bin")).to(device=device, dtype=dtype)
    weight = load_tensor(str(cpp_dir / "weight.bin")).to(device=device, dtype=dtype).requires_grad_()

    output = torch.nn.functional.embedding(indices, weight)
    loss = torch.mean((output - target) ** 2)
    loss.backward()

    cpp_output = load_tensor(str(cpp_dir / "output.bin")).to(device=device, dtype=dtype)
    cpp_grad = load_tensor(str(cpp_dir / "weight_grad.bin")).to(device=device, dtype=dtype)

    results = [
        compare_tensors("output", output, cpp_output),
        compare_tensors("weight_grad", weight.grad, cpp_grad),
    ]
    for r in results:
        print_comparison(r)

    print(f"loss_torch={loss.item():.6f}")


if __name__ == "__main__":
    main()
