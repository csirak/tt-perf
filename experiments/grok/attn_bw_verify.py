#!/usr/bin/env python3
"""Verify attention backward against PyTorch."""

import argparse
import json
from pathlib import Path

import torch

from tensor_io import load_tensor, compare_tensors, print_comparison


def load_meta(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def causal_mask(seq: int, device, dtype):
    mask = torch.triu(torch.ones(seq, seq, device=device, dtype=dtype), diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask * torch.tensor(-1e9, dtype=dtype, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention backward verify")
    parser.add_argument("--cpp-dir", type=str, default="experiments/grok/outputs/attn_bw")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cpp_dir = Path(args.cpp_dir)
    meta = load_meta(cpp_dir / "meta.json")

    batch = int(meta["batch"])
    seq = int(meta["seq"])
    dim = int(meta["dim"])
    heads = int(meta["heads"])
    head_dim = dim // heads
    scale = float(meta["scale"])

    device = torch.device(args.device)
    dtype = torch.bfloat16

    q = load_tensor(str(cpp_dir / "q.bin")).to(device=device, dtype=dtype)
    k = load_tensor(str(cpp_dir / "k.bin")).to(device=device, dtype=dtype)
    v = load_tensor(str(cpp_dir / "v.bin")).to(device=device, dtype=dtype)
    target = load_tensor(str(cpp_dir / "target.bin")).to(device=device, dtype=dtype)

    q = q.detach().requires_grad_(True)
    k = k.detach().requires_grad_(True)
    v = v.detach().requires_grad_(True)

    mask = causal_mask(seq, device, dtype)
    qh = q.view(batch, seq, heads, head_dim).transpose(1, 2)
    kh = k.view(batch, seq, heads, head_dim).transpose(1, 2)
    vh = v.view(batch, seq, heads, head_dim).transpose(1, 2)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * torch.tensor(scale, dtype=dtype, device=device)
    scores_masked = (scores + mask).to(dtype)
    weights = torch.softmax(scores_masked, dim=-1).to(dtype)
    out_heads = torch.matmul(weights, vh).to(dtype)
    out = out_heads.transpose(1, 2).contiguous().view(batch, seq, dim)

    loss = torch.mean((out - target) ** 2)
    loss.backward()

    cpp_scores = load_tensor(str(cpp_dir / "scores.bin")).to(device=device, dtype=dtype)
    cpp_weights = load_tensor(str(cpp_dir / "weights.bin")).to(device=device, dtype=dtype)
    cpp_out = load_tensor(str(cpp_dir / "out.bin")).to(device=device, dtype=dtype)
    cpp_dq = load_tensor(str(cpp_dir / "d_q.bin")).to(device=device, dtype=dtype)
    cpp_dk = load_tensor(str(cpp_dir / "d_k.bin")).to(device=device, dtype=dtype)
    cpp_dv = load_tensor(str(cpp_dir / "d_v.bin")).to(device=device, dtype=dtype)

    results = [
        compare_tensors("scores", scores, cpp_scores),
        compare_tensors("weights", weights, cpp_weights),
        compare_tensors("out", out, cpp_out),
        compare_tensors("d_q", q.grad, cpp_dq),
        compare_tensors("d_k", k.grad, cpp_dk),
        compare_tensors("d_v", v.grad, cpp_dv),
    ]
    for r in results:
        print_comparison(r)

    print(f"loss_torch={loss.item():.6f}")


if __name__ == "__main__":
    main()
