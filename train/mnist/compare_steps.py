#!/usr/bin/env python3
"""Compare TTNN C++ vs PyTorch MNIST parity dumps per step."""

import argparse
import json
import math
import struct
from pathlib import Path

import torch
import numpy as np

import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.append(str(REPO_ROOT / "experiments" / "grok"))
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
    parser = argparse.ArgumentParser(description="Compare MNIST parity dumps")
    parser.add_argument("--cpp-dir", type=str, default="train/mnist/outputs/cpp")
    parser.add_argument("--torch-dir", type=str, default="train/mnist/outputs/pytorch")
    parser.add_argument("--out-dir", type=str, default="train/mnist/outputs/compare")
    args = parser.parse_args()

    cpp_dir = Path(args.cpp_dir)
    torch_dir = Path(args.torch_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step_dirs = sorted([p for p in cpp_dir.glob("step_*" ) if p.is_dir()])
    if not step_dirs:
        raise SystemExit(f"No steps found in {cpp_dir}")

    step_results = []
    summary = {}

    for step_dir in step_dirs:
        step_name = step_dir.name
        torch_step = torch_dir / step_name
        if not torch_step.exists():
            raise SystemExit(f"Missing torch step dir: {torch_step}")

        files = sorted([p.name for p in step_dir.glob("*.bin")])
        for name in files:
            cpp_path = step_dir / name
            torch_path = torch_step / name
            if not torch_path.exists():
                raise SystemExit(f"Missing torch file: {torch_path}")

            if name.endswith("_u32.bin"):
                a = load_u32(cpp_path)
                b = load_u32(torch_path)
                mismatches = int((a != b).sum().item())
                total = a.numel()
                step_results.append({
                    "step": step_name,
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
            step_results.append({
                "step": step_name,
                "tensor": name,
                "shape": "x".join(str(s) for s in a.shape),
                **metrics,
                "mismatch_count": 0,
                "mismatch_rate": 0.0,
            })

            key = name
            best = summary.get(key)
            if best is None or metrics["rel_err_l2"] > best["max_rel_err_l2"]:
                summary[key] = {
                    "tensor": name,
                    "max_rel_err_l2": metrics["rel_err_l2"],
                    "step_max_rel_l2": step_name,
                    "max_rel_err_l1": metrics["rel_err_l1"],
                    "step_max_rel_l1": step_name,
                }

    # Write step results
    step_path = out_dir / "compare_steps.tsv"
    with step_path.open("w", encoding="utf-8") as f:
        f.write("step\ttensor\tshape\tabs_err_l2\trel_err_l2\tabs_err_l1\trel_err_l1\tmax_diff\tcos_sim\tmismatch_count\tmismatch_rate\n")
        for row in step_results:
            f.write(
                f"{row['step']}\t{row['tensor']}\t{row['shape']}\t"
                f"{row['abs_err_l2']}\t{row['rel_err_l2']}\t{row['abs_err_l1']}\t{row['rel_err_l1']}\t"
                f"{row['max_diff']}\t{row['cos_sim']}\t{row['mismatch_count']}\t{row['mismatch_rate']}\n"
            )

    summary_path = out_dir / "compare_summary.tsv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("tensor\tmax_rel_err_l2\tstep_max_rel_l2\tmax_rel_err_l1\tstep_max_rel_l1\n")
        for key in sorted(summary.keys()):
            row = summary[key]
            f.write(
                f"{row['tensor']}\t{row['max_rel_err_l2']}\t{row['step_max_rel_l2']}\t"
                f"{row['max_rel_err_l1']}\t{row['step_max_rel_l1']}\n"
            )

    print(f"Wrote {step_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
