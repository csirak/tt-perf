#!/usr/bin/env python3
"""Compare TTNN C++ vs PyTorch GPT2 reimpl parity dumps."""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent / "grok"))
from tensor_io import load_tensor


def load_u32(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        ndim = struct.unpack("<I", f.read(4))[0]
        shape = [struct.unpack("<I", f.read(4))[0] for _ in range(ndim)]
        numel = 1
        for d in shape:
            numel *= d
        data = f.read(numel * 4)
    arr = np.frombuffer(data, dtype=np.uint32).copy()
    return torch.from_numpy(arr).reshape(shape)


def tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    diff = a_f - b_f
    abs_err_l2 = torch.linalg.vector_norm(diff).item()
    ref_l2 = torch.linalg.vector_norm(b_f).item()
    rel_err_l2 = abs_err_l2 / (ref_l2 + 1e-12)
    abs_err_l1 = torch.linalg.vector_norm(diff, ord=1).item()
    ref_l1 = torch.linalg.vector_norm(b_f, ord=1).item()
    rel_err_l1 = abs_err_l1 / (ref_l1 + 1e-12)
    max_diff = diff.abs().max().item()
    dot = torch.dot(a_f, b_f).item()
    denom = (torch.linalg.vector_norm(a_f).item() * torch.linalg.vector_norm(b_f).item())
    cos_sim = dot / denom if denom > 0 else 0.0
    return {
        "abs_err_l2": abs_err_l2,
        "rel_err_l2": rel_err_l2,
        "abs_err_l1": abs_err_l1,
        "rel_err_l1": rel_err_l1,
        "max_diff": max_diff,
        "cos_sim": cos_sim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GPT2 reimpl parity dumps")
    parser.add_argument("--cpp-dir", type=str, default="experiments/gpt2_reimpl/outputs/cpp/step_0000")
    parser.add_argument("--torch-dir", type=str, default="experiments/gpt2_reimpl/outputs/pytorch/step_0000")
    parser.add_argument("--out-dir", type=str, default="experiments/gpt2_reimpl/outputs/compare")
    args = parser.parse_args()

    cpp_dir = Path(args.cpp_dir)
    torch_dir = Path(args.torch_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cpp_dir.exists():
        raise SystemExit(f"Missing C++ dir: {cpp_dir}")
    if not torch_dir.exists():
        raise SystemExit(f"Missing torch dir: {torch_dir}")

    rows = []
    files = sorted([p.name for p in cpp_dir.glob("*.bin")])
    for name in files:
        cpp_path = cpp_dir / name
        torch_path = torch_dir / name
        if not torch_path.exists():
            raise SystemExit(f"Missing torch file: {torch_path}")

        if name.endswith("_u32.bin"):
            a = load_u32(cpp_path)
            b = load_u32(torch_path)
            mismatches = int((a != b).sum().item())
            total = a.numel()
            rows.append({
                "tensor": name,
                "shape": "x".join(str(s) for s in a.shape),
                "abs_err_l2": 0.0,
                "rel_err_l2": 0.0,
                "abs_err_l1": 0.0,
                "rel_err_l1": 0.0,
                "max_diff": 0.0,
                "cos_sim": 1.0 if mismatches == 0 else 0.0,
                "mismatch_count": mismatches,
                "mismatch_rate": mismatches / max(1, total),
            })
            continue

        a = load_tensor(str(cpp_path))
        b = load_tensor(str(torch_path))
        metrics = tensor_metrics(a, b)
        rows.append({
            "tensor": name,
            "shape": "x".join(str(s) for s in a.shape),
            **metrics,
            "mismatch_count": 0,
            "mismatch_rate": 0.0,
        })

    out_path = out_dir / "compare.tsv"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("tensor\tshape\tabs_err_l2\trel_err_l2\tabs_err_l1\trel_err_l1\tmax_diff\tcos_sim\tmismatch_count\tmismatch_rate\n")
        for row in rows:
            f.write(
                f"{row['tensor']}\t{row['shape']}\t{row['abs_err_l2']}\t{row['rel_err_l2']}\t"
                f"{row['abs_err_l1']}\t{row['rel_err_l1']}\t{row['max_diff']}\t{row['cos_sim']}\t"
                f"{row['mismatch_count']}\t{row['mismatch_rate']}\n"
            )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
