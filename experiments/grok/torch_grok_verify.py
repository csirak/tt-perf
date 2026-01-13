#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Manual PyTorch reference for grok GPT-2 to compare per-layer outputs.

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from tensor_io import load_tensor, save_tensor, compare_tensors, print_comparison


def load_meta(meta_path: Path) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def dyt(x, alpha, gamma, beta, dtype):
    alpha_scalar = alpha.reshape(-1)[0].to(dtype)
    out = torch.tanh(x.to(dtype) * alpha_scalar)
    out = out * gamma.to(dtype) + beta.to(dtype)
    return out.to(dtype)


def layer_norm(x, gamma, beta, eps, dtype):
    g = gamma.reshape(-1).float()
    b = beta.reshape(-1).float()
    y = F.layer_norm(x.float(), (x.shape[-1],), g, b, eps)
    return y.to(dtype)

def rms_norm(x, gamma, beta, eps, dtype):
    y = x.float()
    var = y.pow(2).mean(dim=-1, keepdim=True)
    y = y * torch.rsqrt(var + eps)
    if gamma is not None:
        y = y * gamma.float()
    if beta is not None:
        y = y + beta.float()
    return y.to(dtype)


def linear3d(x, weight, bias, dtype):
    return (torch.matmul(x, weight.t()) + bias).to(dtype)


def ffn(x, w1, b1, w2, b2, dtype, approx: str = "tanh"):
    h = linear3d(x, w1, b1, dtype)
    h = F.gelu(h, approximate=approx).to(dtype)
    return linear3d(h, w2, b2, dtype), h


def causal_mask(seq, dtype):
    return torch.triu(torch.full((1, 1, seq, seq), -1e9, dtype=dtype), diagonal=1)


def save(name, tensor, out_dir):
    save_tensor(tensor, str(out_dir / name))

def save_grad(name, tensor, out_dir, dtype):
    if tensor.grad is None:
        return
    save_tensor(tensor.grad.to(dtype), str(out_dir / name))


def main():
    parser = argparse.ArgumentParser(description="PyTorch grok verification")
    parser.add_argument("--cpp-dir", type=str, default="experiments/grok/outputs/cpp")
    parser.add_argument("--out-dir", type=str, default="experiments/grok/outputs/pytorch")
    parser.add_argument("--gelu-approx", type=str, default="tanh", choices=["tanh", "none"])
    args = parser.parse_args()

    cpp_dir = Path(args.cpp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(cpp_dir / "meta.json")
    batch = int(meta["batch_size"])
    seq = int(meta["pad_to"])
    dim = int(meta["dim"])
    heads = int(meta["heads"])
    ffn_mult = int(meta["ffn_mult"])
    layers = int(meta["layers"])
    vocab = int(meta["vocab"])
    answer_pos = int(meta["answer_pos"])
    loss_scale = float(meta.get("loss_scale", 1.0))
    use_layer_norm = int(meta.get("use_layer_norm", 0))
    use_rms_norm = int(meta.get("use_rms_norm", 0))
    ln_eps = float(meta.get("ln_eps", 1e-5))

    dtype = torch.bfloat16
    head_dim = dim // heads
    scale = 1.0 / head_dim ** 0.5

    tokens = load_tensor(str(cpp_dir / "tokens.bin")).to(dtype=torch.int64)
    tokens = tokens.view(batch, seq)
    targets = load_tensor(str(cpp_dir / "targets.bin")).to(dtype=torch.int64)
    targets = targets.view(batch)

    tok_weight = load_tensor(str(cpp_dir / "tok_weight.bin")).to(dtype).requires_grad_()
    pos_weight = load_tensor(str(cpp_dir / "pos_weight.bin")).to(dtype).requires_grad_()

    tok_embed = F.embedding(tokens, tok_weight)
    pos_indices = torch.arange(seq, dtype=torch.int64).view(1, seq).expand(batch, seq)
    pos_embed = F.embedding(pos_indices, pos_weight)
    tok_plus_pos = (tok_embed + pos_embed).to(dtype)
    tok_plus_pos.retain_grad()
    save("tok_plus_pos.bin", tok_plus_pos, out_dir)

    h = tok_plus_pos
    mask = causal_mask(seq, dtype)
    params = [tok_weight, pos_weight]
    param_tensors = {
        "tok_weight": tok_weight,
        "pos_weight": pos_weight,
    }
    ln_final_gamma = None
    ln_final_beta = None
    if use_rms_norm or use_layer_norm:
        ln_final_gamma = load_tensor(str(cpp_dir / "ln_final_gamma.bin")).to(dtype).requires_grad_()
        ln_final_beta = load_tensor(str(cpp_dir / "ln_final_beta.bin")).to(dtype).requires_grad_()
        params.extend([ln_final_gamma, ln_final_beta])
        param_tensors.update({
            "ln_final_gamma": ln_final_gamma,
            "ln_final_beta": ln_final_beta,
        })

    for layer in range(layers):
        prefix = f"layer{layer}"

        if use_rms_norm or use_layer_norm:
            ln1_gamma = load_tensor(str(cpp_dir / f"{prefix}_ln1_gamma.bin")).to(dtype).requires_grad_()
            ln1_beta = load_tensor(str(cpp_dir / f"{prefix}_ln1_beta.bin")).to(dtype).requires_grad_()
            ln2_gamma = load_tensor(str(cpp_dir / f"{prefix}_ln2_gamma.bin")).to(dtype).requires_grad_()
            ln2_beta = load_tensor(str(cpp_dir / f"{prefix}_ln2_beta.bin")).to(dtype).requires_grad_()
        else:
            ln1_alpha = load_tensor(str(cpp_dir / f"{prefix}_ln1_alpha.bin")).to(dtype).requires_grad_()
            ln1_gamma = load_tensor(str(cpp_dir / f"{prefix}_ln1_gamma.bin")).to(dtype).requires_grad_()
            ln1_beta = load_tensor(str(cpp_dir / f"{prefix}_ln1_beta.bin")).to(dtype).requires_grad_()
            ln2_alpha = load_tensor(str(cpp_dir / f"{prefix}_ln2_alpha.bin")).to(dtype).requires_grad_()
            ln2_gamma = load_tensor(str(cpp_dir / f"{prefix}_ln2_gamma.bin")).to(dtype).requires_grad_()
            ln2_beta = load_tensor(str(cpp_dir / f"{prefix}_ln2_beta.bin")).to(dtype).requires_grad_()

        wq_w = load_tensor(str(cpp_dir / f"{prefix}_wq_weight.bin")).to(dtype).requires_grad_()
        wq_b = load_tensor(str(cpp_dir / f"{prefix}_wq_bias.bin")).to(dtype).requires_grad_()
        wk_w = load_tensor(str(cpp_dir / f"{prefix}_wk_weight.bin")).to(dtype).requires_grad_()
        wk_b = load_tensor(str(cpp_dir / f"{prefix}_wk_bias.bin")).to(dtype).requires_grad_()
        wv_w = load_tensor(str(cpp_dir / f"{prefix}_wv_weight.bin")).to(dtype).requires_grad_()
        wv_b = load_tensor(str(cpp_dir / f"{prefix}_wv_bias.bin")).to(dtype).requires_grad_()
        wo_w = load_tensor(str(cpp_dir / f"{prefix}_wo_weight.bin")).to(dtype).requires_grad_()
        wo_b = load_tensor(str(cpp_dir / f"{prefix}_wo_bias.bin")).to(dtype).requires_grad_()

        ffn_w1 = load_tensor(str(cpp_dir / f"{prefix}_ffn_w1_weight.bin")).to(dtype).requires_grad_()
        ffn_b1 = load_tensor(str(cpp_dir / f"{prefix}_ffn_w1_bias.bin")).to(dtype).requires_grad_()
        ffn_w2 = load_tensor(str(cpp_dir / f"{prefix}_ffn_w2_weight.bin")).to(dtype).requires_grad_()
        ffn_b2 = load_tensor(str(cpp_dir / f"{prefix}_ffn_w2_bias.bin")).to(dtype).requires_grad_()

        if use_rms_norm or use_layer_norm:
            params.extend([
                ln1_gamma, ln1_beta, ln2_gamma, ln2_beta,
                wq_w, wq_b, wk_w, wk_b, wv_w, wv_b, wo_w, wo_b,
                ffn_w1, ffn_b1, ffn_w2, ffn_b2,
            ])
            param_tensors.update({
                f"{prefix}_ln1_gamma": ln1_gamma,
                f"{prefix}_ln1_beta": ln1_beta,
                f"{prefix}_ln2_gamma": ln2_gamma,
                f"{prefix}_ln2_beta": ln2_beta,
                f"{prefix}_wq_weight": wq_w,
                f"{prefix}_wq_bias": wq_b,
                f"{prefix}_wk_weight": wk_w,
                f"{prefix}_wk_bias": wk_b,
                f"{prefix}_wv_weight": wv_w,
                f"{prefix}_wv_bias": wv_b,
                f"{prefix}_wo_weight": wo_w,
                f"{prefix}_wo_bias": wo_b,
                f"{prefix}_ffn_w1_weight": ffn_w1,
                f"{prefix}_ffn_w1_bias": ffn_b1,
                f"{prefix}_ffn_w2_weight": ffn_w2,
                f"{prefix}_ffn_w2_bias": ffn_b2,
            })
        else:
            params.extend([
                ln1_alpha, ln1_gamma, ln1_beta, ln2_alpha, ln2_gamma, ln2_beta,
                wq_w, wq_b, wk_w, wk_b, wv_w, wv_b, wo_w, wo_b,
                ffn_w1, ffn_b1, ffn_w2, ffn_b2,
            ])
            param_tensors.update({
                f"{prefix}_ln1_alpha": ln1_alpha,
                f"{prefix}_ln1_gamma": ln1_gamma,
                f"{prefix}_ln1_beta": ln1_beta,
                f"{prefix}_ln2_alpha": ln2_alpha,
                f"{prefix}_ln2_gamma": ln2_gamma,
                f"{prefix}_ln2_beta": ln2_beta,
                f"{prefix}_wq_weight": wq_w,
                f"{prefix}_wq_bias": wq_b,
                f"{prefix}_wk_weight": wk_w,
                f"{prefix}_wk_bias": wk_b,
                f"{prefix}_wv_weight": wv_w,
                f"{prefix}_wv_bias": wv_b,
                f"{prefix}_wo_weight": wo_w,
                f"{prefix}_wo_bias": wo_b,
                f"{prefix}_ffn_w1_weight": ffn_w1,
                f"{prefix}_ffn_w1_bias": ffn_b1,
                f"{prefix}_ffn_w2_weight": ffn_w2,
                f"{prefix}_ffn_w2_bias": ffn_b2,
            })

        save(f"{prefix}_input.bin", h, out_dir)
        if not (use_rms_norm or use_layer_norm):
            save(f"{prefix}_ln1_alpha.bin", ln1_alpha, out_dir)
        save(f"{prefix}_ln1_gamma.bin", ln1_gamma, out_dir)
        save(f"{prefix}_ln1_beta.bin", ln1_beta, out_dir)

        if use_rms_norm:
            ln1_out = rms_norm(h, ln1_gamma, ln1_beta, ln_eps, dtype)
        elif use_layer_norm:
            ln1_out = layer_norm(h, ln1_gamma, ln1_beta, ln_eps, dtype)
        else:
            ln1_out = dyt(h, ln1_alpha, ln1_gamma, ln1_beta, dtype)
        save(f"{prefix}_ln1.bin", ln1_out, out_dir)

        q = linear3d(ln1_out, wq_w, wq_b, dtype)
        k = linear3d(ln1_out, wk_w, wk_b, dtype)
        v = linear3d(ln1_out, wv_w, wv_b, dtype)
        save(f"{prefix}_wq.bin", q, out_dir)
        save(f"{prefix}_wk.bin", k, out_dir)
        save(f"{prefix}_wv.bin", v, out_dir)

        qh = q.view(batch, seq, heads, head_dim).transpose(1, 2)
        kh = k.view(batch, seq, heads, head_dim).transpose(1, 2)
        vh = v.view(batch, seq, heads, head_dim).transpose(1, 2)

        scores = (torch.matmul(qh, kh.transpose(-1, -2)) * scale).to(dtype)
        save(f"{prefix}_attn_scores.bin", scores, out_dir)
        scores_masked = (scores + mask).to(dtype)
        scores_max = scores_masked.max(dim=-1, keepdim=True).values.to(dtype)
        scores_centered = (scores_masked - scores_max).to(dtype)
        attn_weights = torch.softmax(scores_centered, dim=-1).to(dtype)
        save(f"{prefix}_attn_weights.bin", attn_weights, out_dir)

        attn_out_heads = torch.matmul(attn_weights, vh).to(dtype)
        attn_out = attn_out_heads.transpose(1, 2).contiguous().view(batch, seq, dim)
        save(f"{prefix}_attn_out.bin", attn_out, out_dir)

        wo_out = linear3d(attn_out, wo_w, wo_b, dtype)
        save(f"{prefix}_wo.bin", wo_out, out_dir)

        residual1 = (h + wo_out).to(dtype)
        save(f"{prefix}_residual1.bin", residual1, out_dir)
        if not (use_rms_norm or use_layer_norm):
            save(f"{prefix}_ln2_alpha.bin", ln2_alpha, out_dir)
        save(f"{prefix}_ln2_gamma.bin", ln2_gamma, out_dir)
        save(f"{prefix}_ln2_beta.bin", ln2_beta, out_dir)

        if use_rms_norm:
            ln2_out = rms_norm(residual1, ln2_gamma, ln2_beta, ln_eps, dtype)
        elif use_layer_norm:
            ln2_out = layer_norm(residual1, ln2_gamma, ln2_beta, ln_eps, dtype)
        else:
            ln2_out = dyt(residual1, ln2_alpha, ln2_gamma, ln2_beta, dtype)
        save(f"{prefix}_ln2.bin", ln2_out, out_dir)

        ffn_out, ffn_gelu = ffn(ln2_out, ffn_w1, ffn_b1, ffn_w2, ffn_b2, dtype, args.gelu_approx)
        save(f"{prefix}_ffn_w1.bin", linear3d(ln2_out, ffn_w1, ffn_b1, dtype), out_dir)
        save(f"{prefix}_ffn_gelu.bin", ffn_gelu, out_dir)
        save(f"{prefix}_ffn_w2.bin", ffn_out, out_dir)

        h = (residual1 + ffn_out).to(dtype)
        save(f"{prefix}_output.bin", h, out_dir)

    if use_rms_norm:
        h = rms_norm(h, ln_final_gamma, ln_final_beta, ln_eps, dtype)
        save("ln_final.bin", h, out_dir)
    elif use_layer_norm:
        h = layer_norm(h, ln_final_gamma, ln_final_beta, ln_eps, dtype)
        save("ln_final.bin", h, out_dir)

    out_w = load_tensor(str(cpp_dir / "output_weight.bin")).to(dtype).requires_grad_()
    out_b = load_tensor(str(cpp_dir / "output_bias.bin")).to(dtype).requires_grad_()
    params.extend([out_w, out_b])
    param_tensors.update({
        "output_weight": out_w,
        "output_bias": out_b,
    })

    logits = linear3d(h, out_w, out_b, dtype)
    save("logits.bin", logits, out_dir)

    logits_last = logits[:, answer_pos, :]
    probs = torch.softmax(logits_last.to(dtype), dim=-1).to(dtype)
    log_probs = torch.log(probs).to(dtype)
    loss = -log_probs[torch.arange(batch), targets].mean()
    save("loss.bin", loss.view(1), out_dir)

    # Backward + step (sanity only)

    (loss * loss_scale).backward()
    save_grad("tok_plus_pos_grad.bin", tok_plus_pos, out_dir, dtype)
    for name, tensor in param_tensors.items():
        save_grad(f"{name}_grad.bin", tensor, out_dir, dtype)
    lr = float(meta.get("lr", 1e-3)) if "lr" in meta else 1e-3
    for p in params:
        if p.grad is None:
            continue
        p.data = p.data - lr * p.grad

    # Compare outputs
    print("=" * 70)
    print("Comparing PyTorch outputs vs C++ dumps")
    print("=" * 70)
    compare_files = ["tok_plus_pos.bin"]
    for layer in range(layers):
        prefix = f"layer{layer}"
        compare_files.extend([
            f"{prefix}_input.bin",
            f"{prefix}_ln1.bin",
            f"{prefix}_wq.bin",
            f"{prefix}_wk.bin",
            f"{prefix}_wv.bin",
            f"{prefix}_attn_scores.bin",
            f"{prefix}_attn_weights.bin",
            f"{prefix}_attn_out.bin",
            f"{prefix}_wo.bin",
            f"{prefix}_residual1.bin",
            f"{prefix}_ln2.bin",
            f"{prefix}_ffn_w1.bin",
            f"{prefix}_ffn_gelu.bin",
            f"{prefix}_ffn_w2.bin",
            f"{prefix}_output.bin",
        ])
    if use_rms_norm or use_layer_norm:
        compare_files.append("ln_final.bin")
    compare_files.extend(["logits.bin", "loss.bin"])

    results = []
    for name in compare_files:
        cpp_path = cpp_dir / name
        torch_path = out_dir / name
        if not cpp_path.exists() or not torch_path.exists():
            print(f"  {name:32s}: SKIP (missing)")
            continue
        cpp_t = load_tensor(str(cpp_path))
        torch_t = load_tensor(str(torch_path))
        if cpp_t.shape != torch_t.shape:
            print(f"  {name:32s}: SHAPE MISMATCH {list(cpp_t.shape)} vs {list(torch_t.shape)}")
            continue
        result = compare_tensors(name.replace(".bin", ""), torch_t, cpp_t)
        print_comparison(result)
        results.append(result)

    if results:
        max_diff = max(r["max_diff"] for r in results)
        mean_diff = sum(r["mean_diff"] for r in results) / len(results)
        worst = max(results, key=lambda r: r["max_diff"])
        print("=" * 70)
        print(f"Worst max_diff: {worst['name']} = {worst['max_diff']:.6f}")
        print(f"Average mean_diff: {mean_diff:.6f}")
        print(f"Overall max_diff: {max_diff:.6f}")


if __name__ == "__main__":
    main()
