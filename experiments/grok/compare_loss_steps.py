#!/usr/bin/env python3
# Compare per-step losses between C++ and PyTorch using identical batches.

import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from tensor_io import load_tensor


def load_meta(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
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


def linear3d(x, weight, bias, dtype):
    return (torch.matmul(x, weight.t()) + bias).to(dtype)


def ffn(x, w1, b1, w2, b2, dtype, approx: str = "tanh"):
    h = linear3d(x, w1, b1, dtype)
    h = F.gelu(h, approximate=approx).to(dtype)
    return linear3d(h, w2, b2, dtype)


def causal_mask(seq, dtype):
    mask = torch.triu(torch.full((seq, seq), -1e9, dtype=dtype), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)


def parse_cpp_loss(path: Path) -> dict:
    losses = {}
    if not path.exists():
        return losses
    for line in path.read_text().strip().splitlines():
        if line.startswith("step"):
            continue
        step_str, loss_str = line.split("\t")
        losses[int(step_str)] = float(loss_str)
    return losses


def l2_l1(t: torch.Tensor):
    x = t.detach().float()
    l2 = float(torch.linalg.norm(x))
    l1 = float(torch.sum(torch.abs(x)))
    return l2, l1


def append_act_row(path: Path, header: list[str], row: list):
    write_header = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(header) + "\n")
        f.write("\t".join(str(v) for v in row) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare 10-step loss vs C++")
    parser.add_argument("--cpp-dir", type=str, default="experiments/grok/outputs/cpp")
    parser.add_argument("--steps-dir", type=str, default="experiments/grok/outputs/steps")
    parser.add_argument("--out-path", type=str, default="experiments/grok/outputs/steps/loss_compare.tsv")
    parser.add_argument("--act-out", type=str, default="")
    parser.add_argument("--gelu-approx", type=str, default="tanh", choices=["tanh", "none"])
    args = parser.parse_args()

    cpp_dir = Path(args.cpp_dir)
    steps_dir = Path(args.steps_dir)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    act_out = Path(args.act_out) if args.act_out else (steps_dir / "act_norms_torch.tsv")
    if act_out.exists():
        act_out.unlink()

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
    lr = float(meta.get("lr", 1e-3))
    use_layer_norm = int(meta.get("use_layer_norm", 0))
    ln_eps = float(meta.get("ln_eps", 1e-5))

    dtype = torch.bfloat16
    head_dim = dim // heads
    scale = 1.0 / head_dim ** 0.5

    tok_weight = load_tensor(str(cpp_dir / "tok_weight.bin")).to(dtype).requires_grad_()
    pos_weight = load_tensor(str(cpp_dir / "pos_weight.bin")).to(dtype).requires_grad_()

    params = [tok_weight, pos_weight]
    layer_params = []
    ln_final_gamma = None
    ln_final_beta = None
    if use_layer_norm:
        ln_final_gamma = load_tensor(str(cpp_dir / "ln_final_gamma.bin")).to(dtype).requires_grad_()
        ln_final_beta = load_tensor(str(cpp_dir / "ln_final_beta.bin")).to(dtype).requires_grad_()
        params.extend([ln_final_gamma, ln_final_beta])
    for layer in range(layers):
        prefix = f"layer{layer}"
        if use_layer_norm:
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

        if use_layer_norm:
            params.extend([
                ln1_gamma, ln1_beta, ln2_gamma, ln2_beta,
                wq_w, wq_b, wk_w, wk_b, wv_w, wv_b, wo_w, wo_b,
                ffn_w1, ffn_b1, ffn_w2, ffn_b2,
            ])
            layer_params.append((
                ln1_gamma, ln1_beta, ln2_gamma, ln2_beta,
                wq_w, wq_b, wk_w, wk_b, wv_w, wv_b, wo_w, wo_b,
                ffn_w1, ffn_b1, ffn_w2, ffn_b2,
            ))
        else:
            params.extend([
                ln1_alpha, ln1_gamma, ln1_beta, ln2_alpha, ln2_gamma, ln2_beta,
                wq_w, wq_b, wk_w, wk_b, wv_w, wv_b, wo_w, wo_b,
                ffn_w1, ffn_b1, ffn_w2, ffn_b2,
            ])
            layer_params.append((
                ln1_alpha, ln1_gamma, ln1_beta, ln2_alpha, ln2_gamma, ln2_beta,
                wq_w, wq_b, wk_w, wk_b, wv_w, wv_b, wo_w, wo_b,
                ffn_w1, ffn_b1, ffn_w2, ffn_b2,
            ))

    out_w = load_tensor(str(cpp_dir / "output_weight.bin")).to(dtype).requires_grad_()
    out_b = load_tensor(str(cpp_dir / "output_bias.bin")).to(dtype).requires_grad_()
    params.extend([out_w, out_b])

    mask = causal_mask(seq, dtype)

    step_tokens = sorted(steps_dir.glob("step_*_tokens.bin"))
    step_ids = [int(p.stem.split("_")[1]) for p in step_tokens]
    step_ids.sort()

    cpp_losses = parse_cpp_loss(steps_dir / "loss_cpp.tsv")
    if cpp_losses:
        step_ids = [step for step in step_ids if step in cpp_losses]

    header = ["step", "tok_plus_pos_l2", "tok_plus_pos_l1"]
    for i in range(layers):
        header += [
            f"layer{i}_ln1_l2", f"layer{i}_ln1_l1",
            f"layer{i}_attn_out_l2", f"layer{i}_attn_out_l1",
            f"layer{i}_ln2_l2", f"layer{i}_ln2_l1",
            f"layer{i}_ffn_w2_l2", f"layer{i}_ffn_w2_l1",
            f"layer{i}_output_l2", f"layer{i}_output_l1",
        ]
    header += ["logits_l2", "logits_l1"]

    torch_losses = {}
    torch_grad_norms = {}
    torch_grad_norms_fp32 = {}
    for step in step_ids:
        tokens = load_tensor(str(steps_dir / f"step_{step:04d}_tokens.bin")).to(dtype=torch.int64)
        targets = load_tensor(str(steps_dir / f"step_{step:04d}_targets.bin")).to(dtype=torch.int64)
        tokens = tokens.view(batch, seq)
        targets = targets.view(batch)

        tok_embed = F.embedding(tokens, tok_weight)
        pos_indices = torch.arange(seq, dtype=torch.int64).view(1, seq).expand(batch, seq)
        pos_embed = F.embedding(pos_indices, pos_weight)
        h = (tok_embed + pos_embed).to(dtype)
        row = [step, *l2_l1(h)]

        for params_tuple in layer_params:
            if use_layer_norm:
                (ln1_gamma, ln1_beta, ln2_gamma, ln2_beta,
                 wq_w, wq_b, wk_w, wk_b, wv_w, wv_b, wo_w, wo_b,
                 ffn_w1, ffn_b1, ffn_w2, ffn_b2) = params_tuple
                ln1_out = layer_norm(h, ln1_gamma, ln1_beta, ln_eps, dtype)
            else:
                (ln1_alpha, ln1_gamma, ln1_beta, ln2_alpha, ln2_gamma, ln2_beta,
                 wq_w, wq_b, wk_w, wk_b, wv_w, wv_b, wo_w, wo_b,
                 ffn_w1, ffn_b1, ffn_w2, ffn_b2) = params_tuple
                ln1_out = dyt(h, ln1_alpha, ln1_gamma, ln1_beta, dtype)
            row += [*l2_l1(ln1_out)]
            q = linear3d(ln1_out, wq_w, wq_b, dtype)
            k = linear3d(ln1_out, wk_w, wk_b, dtype)
            v = linear3d(ln1_out, wv_w, wv_b, dtype)
            qh = q.view(batch, seq, heads, head_dim).transpose(1, 2)
            kh = k.view(batch, seq, heads, head_dim).transpose(1, 2)
            vh = v.view(batch, seq, heads, head_dim).transpose(1, 2)
            scores = torch.matmul(qh, kh.transpose(-1, -2)) * torch.tensor(scale, dtype=dtype)
            scores_masked = (scores + mask).to(dtype)
            scores_max = scores_masked.max(dim=-1, keepdim=True).values.to(dtype)
            scores_centered = (scores_masked - scores_max).to(dtype)
            attn_weights = torch.softmax(scores_centered, dim=-1).to(dtype)
            attn_out_heads = torch.matmul(attn_weights, vh).to(dtype)
            attn_out = attn_out_heads.transpose(1, 2).contiguous().view(batch, seq, dim)
            row += [*l2_l1(attn_out)]
            wo_out = linear3d(attn_out, wo_w, wo_b, dtype)
            residual1 = (h + wo_out).to(dtype)
            if use_layer_norm:
                ln2_out = layer_norm(residual1, ln2_gamma, ln2_beta, ln_eps, dtype)
            else:
                ln2_out = dyt(residual1, ln2_alpha, ln2_gamma, ln2_beta, dtype)
            row += [*l2_l1(ln2_out)]
            ffn_out = ffn(ln2_out, ffn_w1, ffn_b1, ffn_w2, ffn_b2, dtype, args.gelu_approx)
            row += [*l2_l1(ffn_out)]
            h = (residual1 + ffn_out).to(dtype)
            row += [*l2_l1(h)]

        if use_layer_norm:
            h = layer_norm(h, ln_final_gamma, ln_final_beta, ln_eps, dtype)
        logits = linear3d(h, out_w, out_b, dtype)
        row += [*l2_l1(logits)]
        append_act_row(act_out, header, row)
        logits_last = logits[:, answer_pos, :]
        probs = torch.softmax(logits_last.to(dtype), dim=-1).to(dtype)
        log_probs = torch.log(probs).to(dtype)
        loss = -log_probs[torch.arange(batch), targets].mean()
        torch_losses[step] = float(loss.detach())

        (loss * loss_scale).backward()
        grad_sum = 0.0
        grad_sum_fp32 = 0.0
        for p in params:
            if p.grad is None:
                continue
            g = p.grad.detach()
            g_fp32 = g.float()
            grad_sum_fp32 += float(torch.sum(g_fp32 * g_fp32))
            g_bf16 = g.to(dtype).float()
            grad_sum += float(torch.sum(g_bf16 * g_bf16))
        torch_grad_norms[step] = float(torch.sqrt(torch.tensor(grad_sum)))
        torch_grad_norms_fp32[step] = float(torch.sqrt(torch.tensor(grad_sum_fp32)))

        for p in params:
            if p.grad is None:
                continue
            p.data = p.data - lr * p.grad
            p.grad = None

    cpp_grad_norms = parse_cpp_loss(steps_dir / "grad_norm_cpp.tsv")

    lines = [
        "step\tloss_cpp\tloss_torch\tabs_diff\trel_diff"
        "\tgrad_norm_cpp\tgrad_norm_torch_bf16\tgrad_norm_abs_diff\tgrad_norm_rel_diff"
        "\tgrad_norm_torch_fp32\tgrad_norm_fp32_abs_diff\tgrad_norm_fp32_rel_diff"
    ]
    for step in step_ids:
        loss_cpp = cpp_losses.get(step, float("nan"))
        loss_torch = torch_losses.get(step, float("nan"))
        abs_diff = abs(loss_cpp - loss_torch) if (loss_cpp == loss_cpp and loss_torch == loss_torch) else float("nan")
        rel_diff = abs_diff / abs(loss_cpp) if (loss_cpp == loss_cpp and loss_cpp != 0.0) else float("nan")
        grad_cpp = cpp_grad_norms.get(step, float("nan"))
        grad_torch = torch_grad_norms.get(step, float("nan"))
        grad_abs = abs(grad_cpp - grad_torch) if (grad_cpp == grad_cpp and grad_torch == grad_torch) else float("nan")
        grad_rel = grad_abs / abs(grad_cpp) if (grad_cpp == grad_cpp and grad_cpp != 0.0) else float("nan")
        grad_torch_fp32 = torch_grad_norms_fp32.get(step, float("nan"))
        grad_abs_fp32 = abs(grad_cpp - grad_torch_fp32) if (grad_cpp == grad_cpp and grad_torch_fp32 == grad_torch_fp32) else float("nan")
        grad_rel_fp32 = grad_abs_fp32 / abs(grad_cpp) if (grad_cpp == grad_cpp and grad_cpp != 0.0) else float("nan")
        lines.append(
            f"{step}\t{loss_cpp:.6g}\t{loss_torch:.6g}\t{abs_diff:.6g}\t{rel_diff:.6g}"
            f"\t{grad_cpp:.6g}\t{grad_torch:.6g}\t{grad_abs:.6g}\t{grad_rel:.6g}"
            f"\t{grad_torch_fp32:.6g}\t{grad_abs_fp32:.6g}\t{grad_rel_fp32:.6g}"
        )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
