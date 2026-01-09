#!/usr/bin/env python3
"""Verify AdamW update against C++ step weights using step_0000 grads."""

import argparse
import math
from pathlib import Path

import torch

import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent / "grok"))
from tensor_io import load_tensor


def load_step_tensor(step_dir: Path, name: str) -> torch.Tensor:
    return load_tensor(str(step_dir / name)).to(torch.bfloat16)


def adamw_step(w, g, lr, beta1, beta2, eps, wd, apply_wd: bool):
    # Simulate C++ Adam: m,v initialized to 0, bf16 ops for m,v and update, fp32 master.
    m = (1.0 - beta1) * g
    v = (1.0 - beta2) * (g * g)
    step_size = lr * math.sqrt(1.0 - beta2) / (1.0 - beta1)
    update = (m / (torch.sqrt(v) + eps)) * step_size

    w_fp32 = w.float()
    if apply_wd and wd > 0.0:
        w_fp32 = w_fp32 * (1.0 - lr * wd) - update.float()
    else:
        w_fp32 = w_fp32 - update.float()

    return w_fp32.to(torch.bfloat16)


def compare(name: str, expected: torch.Tensor, actual: torch.Tensor):
    diff = (expected.float() - actual.float()).reshape(-1)
    abs_l2 = torch.linalg.vector_norm(diff).item()
    rel_l2 = abs_l2 / (torch.linalg.vector_norm(actual.float().reshape(-1)).item() + 1e-12)
    max_diff = diff.abs().max().item()
    print(f"{name}: rel_l2={rel_l2:.6f} max_diff={max_diff:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Check AdamW update")
    parser.add_argument("--cpp-dir", type=str, default="experiments/mnist/outputs/cpp")
    parser.add_argument("--config", type=str, default=str(THIS_DIR / "default.yaml"))
    args = parser.parse_args()

    # Load config manually
    kv = {}
    for raw in Path(args.config).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        kv[k.strip()] = v.strip()

    lr = float(kv.get("lr", 1e-3))
    beta1 = float(kv.get("beta1", 0.9))
    beta2 = float(kv.get("beta2", 0.999))
    eps = float(kv.get("eps", 1e-8))
    wd = float(kv.get("weight_decay", 0.0))

    cpp_dir = Path(args.cpp_dir)
    step0 = cpp_dir / "step_0000"
    step1 = cpp_dir / "step_0001"

    if not step0.exists() or not step1.exists():
        raise SystemExit("Need step_0000 and step_0001 in cpp dir")

    w1 = load_step_tensor(step0, "w1.bin")
    b1 = load_step_tensor(step0, "b1.bin")
    w2 = load_step_tensor(step0, "w2.bin")
    b2 = load_step_tensor(step0, "b2.bin")

    w1g = load_step_tensor(step0, "w1_grad.bin")
    b1g = load_step_tensor(step0, "b1_grad.bin")
    w2g = load_step_tensor(step0, "w2_grad.bin")
    b2g = load_step_tensor(step0, "b2_grad.bin")

    w1_next = load_step_tensor(step1, "w1.bin")
    b1_next = load_step_tensor(step1, "b1.bin")
    w2_next = load_step_tensor(step1, "w2.bin")
    b2_next = load_step_tensor(step1, "b2.bin")

    w1_exp = adamw_step(w1, w1g, lr, beta1, beta2, eps, wd, True)
    b1_exp = adamw_step(b1, b1g, lr, beta1, beta2, eps, wd, False)
    w2_exp = adamw_step(w2, w2g, lr, beta1, beta2, eps, wd, True)
    b2_exp = adamw_step(b2, b2g, lr, beta1, beta2, eps, wd, False)

    compare("w1", w1_exp, w1_next)
    compare("b1", b1_exp, b1_next)
    compare("w2", w2_exp, w2_next)
    compare("b2", b2_exp, b2_next)


if __name__ == "__main__":
    main()
