#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tensordiff.td_io import load_tensor, save_tensor  # noqa: E402


def load(name: str, out_dir: Path) -> torch.Tensor:
    return load_tensor(str(out_dir / name)).to(torch.float32)


def save(name: str, t: torch.Tensor, out_dir: Path) -> None:
    save_tensor(t, str(out_dir / name), origin="torch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="tensordiffgrad output dir")
    args = ap.parse_args()

    out_dir = Path(args.dir)
    norm_kind = (os.getenv("TENSORDIFFGRAD_NORM") or "rmsnorm").lower()

    x = load("input.bin", out_dir).requires_grad_(True)
    bsz, seq, dim = x.shape
    ffn_dim = load("w1.bin", out_dir).shape[1]

    # Params
    wq = load("wq.bin", out_dir).requires_grad_(True)
    wk = load("wk.bin", out_dir).requires_grad_(True)
    wv = load("wv.bin", out_dir).requires_grad_(True)
    wo = load("wo.bin", out_dir).requires_grad_(True)
    w1 = load("w1.bin", out_dir).requires_grad_(True)
    w2 = load("w2.bin", out_dir).requires_grad_(True)

    bq = load("bq.bin", out_dir).requires_grad_(True)
    bk = load("bk.bin", out_dir).requires_grad_(True)
    bv = load("bv.bin", out_dir).requires_grad_(True)
    bo = load("bo.bin", out_dir).requires_grad_(True)
    b1 = load("b1.bin", out_dir).requires_grad_(True)
    b2 = load("b2.bin", out_dir).requires_grad_(True)

    ln1_gamma = load("ln1_gamma.bin", out_dir).requires_grad_(True)
    ln1_beta = load("ln1_beta.bin", out_dir).requires_grad_(True)
    ln2_gamma = load("ln2_gamma.bin", out_dir).requires_grad_(True)
    ln2_beta = load("ln2_beta.bin", out_dir).requires_grad_(True)

    ln1_alpha = None
    ln2_alpha = None
    if (out_dir / "ln1_alpha.bin").exists():
        ln1_alpha = load("ln1_alpha.bin", out_dir).requires_grad_(True)
    if (out_dir / "ln2_alpha.bin").exists():
        ln2_alpha = load("ln2_alpha.bin", out_dir).requires_grad_(True)

    def apply_norm(x, gamma, beta, alpha):
        if norm_kind == "layernorm":
            return torch.nn.functional.layer_norm(x, (dim,), gamma.view(-1), beta.view(-1), eps=1e-5)
        if norm_kind == "dyt":
            t = torch.tanh(alpha.view(1, 1, 1) * x)
            return t * gamma + beta
        # rmsnorm
        mean_sq = (x * x).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(mean_sq + 1e-5)
        x_norm = x * rstd
        return x_norm * gamma + beta

    ln1 = apply_norm(x, ln1_gamma, ln1_beta, ln1_alpha)
    q = torch.matmul(ln1, wq) + bq
    k = torch.matmul(ln1, wk) + bk
    v = torch.matmul(ln1, wv) + bv

    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / (dim ** 0.5))

    # Causal mask
    mask = torch.triu(torch.ones((seq, seq), dtype=torch.float32), diagonal=1) * -1e4
    scores = scores + mask

    attn = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn, v)
    attn_proj = torch.matmul(attn_out, wo) + bo
    residual1 = x + attn_proj

    ln2 = apply_norm(residual1, ln2_gamma, ln2_beta, ln2_alpha)
    ffn1 = torch.matmul(ln2, w1) + b1
    ffn_gelu = torch.nn.functional.gelu(ffn1, approximate="tanh")
    ffn2 = torch.matmul(ffn_gelu, w2) + b2
    out = residual1 + ffn2
    out.retain_grad()

    loss = out.sum()
    loss.backward()

    # Save
    save("torch_output.bin", out, out_dir)
    save("torch_loss.bin", loss.view(1), out_dir)
    save("torch_input_grad.bin", x.grad, out_dir)
    save("torch_output_grad.bin", out.grad, out_dir)

    save("torch_wq_grad.bin", wq.grad, out_dir)
    save("torch_wk_grad.bin", wk.grad, out_dir)
    save("torch_wv_grad.bin", wv.grad, out_dir)
    save("torch_wo_grad.bin", wo.grad, out_dir)
    save("torch_w1_grad.bin", w1.grad, out_dir)
    save("torch_w2_grad.bin", w2.grad, out_dir)
    save("torch_bq_grad.bin", bq.grad, out_dir)
    save("torch_bk_grad.bin", bk.grad, out_dir)
    save("torch_bv_grad.bin", bv.grad, out_dir)
    save("torch_bo_grad.bin", bo.grad, out_dir)
    save("torch_b1_grad.bin", b1.grad, out_dir)
    save("torch_b2_grad.bin", b2.grad, out_dir)
    save("torch_ln1_gamma_grad.bin", ln1_gamma.grad, out_dir)
    save("torch_ln1_beta_grad.bin", ln1_beta.grad, out_dir)
    save("torch_ln2_gamma_grad.bin", ln2_gamma.grad, out_dir)
    save("torch_ln2_beta_grad.bin", ln2_beta.grad, out_dir)
    if ln1_alpha is not None and ln1_alpha.grad is not None:
        save("torch_ln1_alpha_grad.bin", ln1_alpha.grad, out_dir)
    if ln2_alpha is not None and ln2_alpha.grad is not None:
        save("torch_ln2_alpha_grad.bin", ln2_alpha.grad, out_dir)

    print(f"torch ref done: {out_dir}")


if __name__ == "__main__":
    main()
