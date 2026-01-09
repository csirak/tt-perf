#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch

from tensor_io import load_tensor, compare_tensors, print_comparison


def load_meta(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def causal_mask(seq: int, device, dtype):
    mask = torch.triu(torch.ones(seq, seq, device=device, dtype=dtype), diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask * torch.tensor(-1e9, dtype=dtype, device=device)


def main():
    parser = argparse.ArgumentParser(description="Compare attention math using C++ Q/K/V dumps")
    parser.add_argument("--cpp-dir", type=str, default="experiments/grok/outputs/cpp")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cpp_dir = Path(args.cpp_dir)
    meta = load_meta(cpp_dir / "meta.json")

    dim = int(meta["dim"])
    heads = int(meta["heads"])
    head_dim = dim // heads
    scale = 1.0 / math.sqrt(head_dim)

    prefix = f"layer{args.layer}"

    q = load_tensor(str(cpp_dir / f"{prefix}_wq.bin"))
    k = load_tensor(str(cpp_dir / f"{prefix}_wk.bin"))
    v = load_tensor(str(cpp_dir / f"{prefix}_wv.bin"))

    cpp_scores = load_tensor(str(cpp_dir / f"{prefix}_attn_scores.bin"))
    cpp_weights = load_tensor(str(cpp_dir / f"{prefix}_attn_weights.bin"))
    cpp_out = load_tensor(str(cpp_dir / f"{prefix}_attn_out.bin"))

    device = torch.device(args.device)
    dtype = torch.bfloat16

    q = q.to(device=device, dtype=dtype)
    k = k.to(device=device, dtype=dtype)
    v = v.to(device=device, dtype=dtype)
    cpp_scores = cpp_scores.to(device=device, dtype=dtype)
    cpp_weights = cpp_weights.to(device=device, dtype=dtype)
    cpp_out = cpp_out.to(device=device, dtype=dtype)

    B, S, D = q.shape
    mask = causal_mask(S, device, dtype)

    # Compute scores in BF16 (multi-head)
    qh = q.view(B, S, heads, head_dim).transpose(1, 2)
    kh = k.view(B, S, heads, head_dim).transpose(1, 2)
    vh = v.view(B, S, heads, head_dim).transpose(1, 2)

    scores = torch.matmul(qh, kh.transpose(-1, -2)) * torch.tensor(scale, dtype=dtype, device=device)
    scores_masked = scores + mask
    weights = torch.softmax(scores_masked, dim=-1)
    out_heads = torch.matmul(weights, vh)
    out = out_heads.transpose(1, 2).contiguous().view(B, S, D)

    print("=" * 70)
    print(f"Attention compare: layer={args.layer}, scale={scale:.6f}, shape B={B} S={S} D={D}")
    print("=" * 70)

    # Compare BF16 computations
    results = []
    results.append(compare_tensors("attn_scores", scores, cpp_scores))
    results.append(compare_tensors("attn_weights", weights, cpp_weights))
    results.append(compare_tensors("attn_out", out, cpp_out))
    # Isolate matmul using C++ weights
    out_from_cpp_weights = torch.matmul(cpp_weights, vh)
    out_from_cpp_weights = out_from_cpp_weights.transpose(1, 2).contiguous().view(B, S, D)
    results.append(compare_tensors("attn_out_cppW", out_from_cpp_weights, cpp_out))
    for r in results:
        print_comparison(r)

    # Also compute in FP32 for reference
    scores_f = torch.matmul(qh.float(), kh.float().transpose(-1, -2)) * scale
    scores_f_masked = scores_f + mask.float()
    weights_f = torch.softmax(scores_f_masked, dim=-1)
    out_f = torch.matmul(weights_f, vh.float())
    out_f = out_f.transpose(1, 2).contiguous().view(B, S, D)

    out_cppW_f = torch.matmul(cpp_weights.float(), vh.float())
    out_cppW_f = out_cppW_f.transpose(1, 2).contiguous().view(B, S, D)

    print("=" * 70)
    print("FP32 reference vs C++ BF16 (cast to FP32)")
    print("=" * 70)
    results_fp32 = []
    results_fp32.append(compare_tensors("attn_scores_fp32", scores_f, cpp_scores.float()))
    results_fp32.append(compare_tensors("attn_weights_fp32", weights_f, cpp_weights.float()))
    results_fp32.append(compare_tensors("attn_out_fp32", out_f, cpp_out.float()))
    results_fp32.append(compare_tensors("attn_out_cppW_fp32", out_cppW_f, cpp_out.float()))
    for r in results_fp32:
        print_comparison(r)


if __name__ == "__main__":
    main()
