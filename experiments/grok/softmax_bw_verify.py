#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from tensor_io import load_tensor, compare_tensors, print_comparison


def load_meta(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Verify softmax backward against PyTorch")
    parser.add_argument("--cpp-dir", type=str, default="experiments/grok/outputs/softmax_bw")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cpp_dir = Path(args.cpp_dir)
    meta = load_meta(cpp_dir / "meta.json")
    batch = int(meta["batch"])
    seq = int(meta["seq"])

    device = torch.device(args.device)
    dtype = torch.bfloat16

    scores = load_tensor(str(cpp_dir / "scores.bin")).to(device=device, dtype=dtype)
    d_attn = load_tensor(str(cpp_dir / "d_attn.bin")).to(device=device, dtype=dtype)
    weights_cpp = load_tensor(str(cpp_dir / "weights.bin")).to(device=device, dtype=dtype)
    d_scores_neg1 = load_tensor(str(cpp_dir / "d_scores_neg1.bin")).to(device=device, dtype=dtype)
    d_scores_dim2 = load_tensor(str(cpp_dir / "d_scores_dim2.bin")).to(device=device, dtype=dtype)

    scores = scores.view(batch, seq, seq)
    d_attn = d_attn.view(batch, seq, seq)

    # Compute softmax in torch from scores
    weights = torch.softmax(scores.float(), dim=-1).to(dtype)

    # Softmax backward formula
    dy_y = d_attn * weights
    sum_dy_y = dy_y.sum(dim=-1, keepdim=True)
    d_scores = weights * (d_attn - sum_dy_y)

    print("=" * 70)
    print("Softmax backward compare")
    print("=" * 70)

    print("-- Weights compare (torch vs C++)")
    print_comparison(compare_tensors("weights", weights, weights_cpp))

    print("-- d_scores compare")
    print_comparison(compare_tensors("d_scores_neg1", d_scores, d_scores_neg1))
    print_comparison(compare_tensors("d_scores_dim2", d_scores, d_scores_dim2))


if __name__ == "__main__":
    main()
