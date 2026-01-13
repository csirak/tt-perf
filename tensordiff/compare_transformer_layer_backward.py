#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import torch

from td_io import load_tensor


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    af = a.float().flatten()
    bf = b.float().flatten()
    if af.numel() == 0:
        return {"numel": 0, "max_abs": 0.0, "mean_abs": 0.0, "rel_l2": 0.0}
    diff = (af - bf).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = bf.pow(2).sum().sqrt().item()
    rel_l2 = (diff.pow(2).sum().sqrt().item() / denom) if denom != 0 else 0.0
    return {"numel": diff.numel(), "max_abs": max_abs, "mean_abs": mean_abs, "rel_l2": rel_l2}


ORDER = [
    ("d_output", "torch_d_output.bin", "ttnn_d_output.bin"),
    ("d_ffn2", "torch_d_ffn2.bin", "ttnn_d_ffn2.bin"),
    ("d_ffn_gelu", "torch_d_ffn_gelu.bin", "ttnn_d_ffn_gelu.bin"),
    ("d_ffn1", "torch_d_ffn1.bin", "ttnn_d_ffn1.bin"),
    ("d_ln2", "torch_d_ln2.bin", "ttnn_d_ln2.bin"),
    ("d_residual1", "torch_d_residual1.bin", "ttnn_d_residual1.bin"),
    ("d_attn_proj", "torch_d_attn_proj.bin", "ttnn_d_attn_proj.bin"),
    ("d_attn_out", "torch_d_attn_out.bin", "ttnn_d_attn_out.bin"),
    ("d_attn_weights", "torch_d_attn_weights.bin", "ttnn_d_attn_weights.bin"),
    ("d_attn_scores", "torch_d_attn_scores.bin", "ttnn_d_attn_scores.bin"),
    ("d_q", "torch_d_q.bin", "ttnn_d_q.bin"),
    ("d_k", "torch_d_k.bin", "ttnn_d_k.bin"),
    ("d_v", "torch_d_v.bin", "ttnn_d_v.bin"),
    ("d_ln1", "torch_d_ln1.bin", "ttnn_d_ln1.bin"),
    ("d_x", "torch_d_x.bin", "ttnn_d_x.bin"),
]

PARAMS = [
    ("wq_weight_grad", "torch_wq_weight_grad.bin", "ttnn_wq_weight_grad.bin"),
    ("wk_weight_grad", "torch_wk_weight_grad.bin", "ttnn_wk_weight_grad.bin"),
    ("wv_weight_grad", "torch_wv_weight_grad.bin", "ttnn_wv_weight_grad.bin"),
    ("wo_weight_grad", "torch_wo_weight_grad.bin", "ttnn_wo_weight_grad.bin"),
    ("wq_bias_grad", "torch_wq_bias_grad.bin", "ttnn_wq_bias_grad.bin"),
    ("wk_bias_grad", "torch_wk_bias_grad.bin", "ttnn_wk_bias_grad.bin"),
    ("wv_bias_grad", "torch_wv_bias_grad.bin", "ttnn_wv_bias_grad.bin"),
    ("wo_bias_grad", "torch_wo_bias_grad.bin", "ttnn_wo_bias_grad.bin"),
    ("ffn_w1_weight_grad", "torch_ffn_w1_weight_grad.bin", "ttnn_ffn_w1_weight_grad.bin"),
    ("ffn_w1_bias_grad", "torch_ffn_w1_bias_grad.bin", "ttnn_ffn_w1_bias_grad.bin"),
    ("ffn_w2_weight_grad", "torch_ffn_w2_weight_grad.bin", "ttnn_ffn_w2_weight_grad.bin"),
    ("ffn_w2_bias_grad", "torch_ffn_w2_bias_grad.bin", "ttnn_ffn_w2_bias_grad.bin"),
    ("ln1_gamma_grad", "torch_ln1_gamma_grad.bin", "ttnn_ln1_gamma_grad.bin"),
    ("ln1_beta_grad", "torch_ln1_beta_grad.bin", "ttnn_ln1_beta_grad.bin"),
    ("ln2_gamma_grad", "torch_ln2_gamma_grad.bin", "ttnn_ln2_gamma_grad.bin"),
    ("ln2_beta_grad", "torch_ln2_beta_grad.bin", "ttnn_ln2_beta_grad.bin"),
]


def run_list(out_dir: Path, entries, label: str) -> list:
    rows = []
    for name, torch_file, ttnn_file in entries:
        torch_path = out_dir / torch_file
        ttnn_path = out_dir / ttnn_file
        if not torch_path.exists() or not ttnn_path.exists():
            print(f"# missing {label} {name}: {torch_file} or {ttnn_file}")
            continue
        t_torch = load_tensor(torch_path)
        t_ttnn = load_tensor(ttnn_path)
        result = compare(t_torch, t_ttnn)
        rows.append((name, result))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="tensordiff/outputs/transformer_layer_bwd_torch_rand")
    args = parser.parse_args()

    out_dir = Path(args.dir)
    rows = run_list(out_dir, ORDER, "grad")
    params = run_list(out_dir, PARAMS, "param")

    print("# backward_grad_compare")
    print("name\tnumel\tmax_abs\tmean_abs\trel_l2")
    for name, r in rows:
        print(f"{name}\t{r['numel']}\t{r['max_abs']:.6f}\t{r['mean_abs']:.6f}\t{r['rel_l2']:.6f}")

    print("\n# param_grad_compare")
    print("name\tnumel\tmax_abs\tmean_abs\trel_l2")
    for name, r in params:
        print(f"{name}\t{r['numel']}\t{r['max_abs']:.6f}\t{r['mean_abs']:.6f}\t{r['rel_l2']:.6f}")


if __name__ == "__main__":
    main()
