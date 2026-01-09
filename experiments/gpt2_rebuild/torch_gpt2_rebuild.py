#!/usr/bin/env python3
"""PyTorch functional parity for GPT2 rebuild (incremental stages)."""

import argparse
import json
import math
import struct
from pathlib import Path

import torch
import numpy as np

import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent / "grok"))
from tensor_io import load_tensor, save_tensor


def load_kv_file(path: Path) -> dict:
    kv = {}
    if not path.exists():
        return kv
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        kv[key.strip()] = val.strip()
    return kv


def get_int(kv: dict, key: str, default: int) -> int:
    return int(kv.get(key, default))


def get_float(kv: dict, key: str, default: float) -> float:
    return float(kv.get(key, default))


def get_str(kv: dict, key: str, default: str) -> str:
    return str(kv.get(key, default))


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


def save_u32(t: torch.Tensor, path: Path) -> None:
    t = t.detach().cpu().to(torch.uint32).contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<I", t.ndim))
        for dim in t.shape:
            f.write(struct.pack("<I", dim))
        f.write(t.numpy().tobytes())


def matmul_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.matmul(a.float(), b.float())
    return out.to(torch.bfloat16)


def linear3d(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    y = matmul_fp32(x, w.t())
    return (y + b).to(torch.bfloat16)


def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.float()
    mean = x_f.mean(dim=-1, keepdim=True)
    centered = x_f - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    x_norm = centered * rstd
    out = gamma.float() * x_norm + beta.float()
    return out.to(torch.bfloat16)


def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(x.float(), approximate="none").to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT2 rebuild parity (PyTorch)")
    parser.add_argument("--config", type=str, default=str(THIS_DIR / "default.yaml"))
    parser.add_argument("--cpp-dir", type=str, default="experiments/gpt2_rebuild/outputs/cpp/step_0000")
    parser.add_argument("--out-dir", type=str, default="experiments/gpt2_rebuild/outputs/pytorch/step_0000")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    kv = load_kv_file(Path(args.config))
    batch = get_int(kv, "batch_size", 32)
    seq = get_int(kv, "seq", 32)
    dim = get_int(kv, "dim", 128)
    heads = get_int(kv, "heads", 4)
    ffn_mult = get_int(kv, "ffn_mult", 4)
    vocab = get_int(kv, "vocab", 160)
    stage = get_int(kv, "stage", 4)
    ln_eps = get_float(kv, "ln_eps", 1e-5)

    if stage < 0 or stage > 4:
        raise ValueError("stage must be in [0,4]")

    head_dim = dim // heads
    device = torch.device(args.device)

    cpp_dir = Path(args.cpp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = load_u32(cpp_dir / "tokens_u32.bin").to(device=device, dtype=torch.long)
    pos = load_u32(cpp_dir / "pos_u32.bin").to(device=device, dtype=torch.long)

    tok_weight = load_tensor(str(cpp_dir / "tok_weight.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()
    pos_weight = load_tensor(str(cpp_dir / "pos_weight.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()

    ln1_gamma = ln1_beta = None
    ln2_gamma = ln2_beta = None
    wq = wq_b = wk = wk_b = wv = wv_b = wo = wo_b = None
    ffn_w1 = ffn_b1 = ffn_w2 = ffn_b2 = None
    lm_head_w = lm_head_b = None

    if stage >= 1:
        ln1_gamma = load_tensor(str(cpp_dir / "ln1_gamma.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()
        ln1_beta = load_tensor(str(cpp_dir / "ln1_beta.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()

    if stage >= 2:
        wq = load_tensor(str(cpp_dir / "wq_weight.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()
        wq_b = load_tensor(str(cpp_dir / "wq_bias.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()
        wk = load_tensor(str(cpp_dir / "wk_weight.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()
        wk_b = load_tensor(str(cpp_dir / "wk_bias.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()
        wv = load_tensor(str(cpp_dir / "wv_weight.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()
        wv_b = load_tensor(str(cpp_dir / "wv_bias.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()
        wo = load_tensor(str(cpp_dir / "wo_weight.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()
        wo_b = load_tensor(str(cpp_dir / "wo_bias.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()

    if stage >= 3:
        ln2_gamma = load_tensor(str(cpp_dir / "ln2_gamma.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()
        ln2_beta = load_tensor(str(cpp_dir / "ln2_beta.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()

        ffn_w1 = load_tensor(str(cpp_dir / "ffn_w1_weight.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()
        ffn_b1 = load_tensor(str(cpp_dir / "ffn_w1_bias.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()
        ffn_w2 = load_tensor(str(cpp_dir / "ffn_w2_weight.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()
        ffn_b2 = load_tensor(str(cpp_dir / "ffn_w2_bias.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()

    if stage >= 4:
        lm_head_w = load_tensor(str(cpp_dir / "lm_head_weight.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()
        lm_head_b = load_tensor(str(cpp_dir / "lm_head_bias.bin")).to(device=device, dtype=torch.bfloat16).requires_grad_()

    target = load_tensor(str(cpp_dir / "target.bin")).to(device=device, dtype=torch.bfloat16)

    tok_embed = torch.nn.functional.embedding(tokens, tok_weight)
    pos_embed = torch.nn.functional.embedding(pos, pos_weight)
    tok_plus_pos = (tok_embed + pos_embed).to(torch.bfloat16)

    ln1_out = None
    q = k = v = None
    q_heads = k_heads = v_heads = None
    scores = scores_masked = None
    attn_weights = None
    attn_out = attn_proj = residual1 = None
    ln2_out = None
    ffn_w1_out = ffn_gelu = ffn_w2_out = None
    output = None
    logits = None

    if stage == 0:
        stage_out = tok_plus_pos
    else:
        ln1_out = layer_norm(tok_plus_pos, ln1_gamma, ln1_beta, ln_eps)
        if stage == 1:
            stage_out = ln1_out
        else:
            q = linear3d(ln1_out, wq, wq_b)
            k = linear3d(ln1_out, wk, wk_b)
            v = linear3d(ln1_out, wv, wv_b)

            q_4d = q.view(batch, seq, heads, head_dim)
            k_4d = k.view(batch, seq, heads, head_dim)
            v_4d = v.view(batch, seq, heads, head_dim)

            q_perm = q_4d.permute(0, 2, 1, 3).contiguous()
            k_perm = k_4d.permute(0, 2, 1, 3).contiguous()
            v_perm = v_4d.permute(0, 2, 1, 3).contiguous()

            q_heads = q_perm.reshape(batch * heads, seq, head_dim)
            k_heads = k_perm.reshape(batch * heads, seq, head_dim)
            v_heads = v_perm.reshape(batch * heads, seq, head_dim)

            scores = matmul_fp32(q_heads, k_heads.transpose(-2, -1))
            scores = (scores.float() * (1.0 / math.sqrt(head_dim))).to(torch.bfloat16)

            mask = torch.full((1, seq, seq), -1e9, device=device, dtype=torch.bfloat16)
            mask = torch.triu(mask, diagonal=1)
            scores_masked = (scores + mask).to(torch.bfloat16)

            attn_weights = torch.softmax(scores_masked.float(), dim=-1).to(torch.bfloat16)
            attn_out_heads = matmul_fp32(attn_weights, v_heads)

            attn_out_4d = attn_out_heads.reshape(batch, heads, seq, head_dim)
            attn_out_perm = attn_out_4d.permute(0, 2, 1, 3).contiguous()
            attn_out = attn_out_perm.reshape(batch, seq, dim)

            attn_proj = linear3d(attn_out, wo, wo_b)
            residual1 = (tok_plus_pos + attn_proj).to(torch.bfloat16)

            if stage == 2:
                stage_out = residual1
            else:
                ln2_out = layer_norm(residual1, ln2_gamma, ln2_beta, ln_eps)
                ffn_w1_out = linear3d(ln2_out, ffn_w1, ffn_b1)
                ffn_gelu = gelu_exact(ffn_w1_out)
                ffn_w2_out = linear3d(ffn_gelu, ffn_w2, ffn_b2)
                output = (residual1 + ffn_w2_out).to(torch.bfloat16)

                if stage == 3:
                    stage_out = output
                else:
                    logits = linear3d(output, lm_head_w, lm_head_b)
                    logits.retain_grad()
                    stage_out = logits

    diff = (stage_out - target).to(torch.bfloat16)
    loss = (diff * diff).mean().to(torch.bfloat16)

    loss.backward()

    dump_ln1 = stage >= 1
    dump_attn = stage >= 2
    dump_ffn = stage >= 3
    dump_logits = stage >= 4

    save_u32(tokens, out_dir / "tokens_u32.bin")
    save_u32(pos, out_dir / "pos_u32.bin")

    save_tensor(tok_weight, out_dir / "tok_weight.bin")
    save_tensor(pos_weight, out_dir / "pos_weight.bin")
    if dump_ln1:
        save_tensor(ln1_gamma, out_dir / "ln1_gamma.bin")
        save_tensor(ln1_beta, out_dir / "ln1_beta.bin")
    if dump_ffn:
        save_tensor(ln2_gamma, out_dir / "ln2_gamma.bin")
        save_tensor(ln2_beta, out_dir / "ln2_beta.bin")

    if dump_attn:
        save_tensor(wq, out_dir / "wq_weight.bin")
        save_tensor(wq_b, out_dir / "wq_bias.bin")
        save_tensor(wk, out_dir / "wk_weight.bin")
        save_tensor(wk_b, out_dir / "wk_bias.bin")
        save_tensor(wv, out_dir / "wv_weight.bin")
        save_tensor(wv_b, out_dir / "wv_bias.bin")
        save_tensor(wo, out_dir / "wo_weight.bin")
        save_tensor(wo_b, out_dir / "wo_bias.bin")

    if dump_ffn:
        save_tensor(ffn_w1, out_dir / "ffn_w1_weight.bin")
        save_tensor(ffn_b1, out_dir / "ffn_w1_bias.bin")
        save_tensor(ffn_w2, out_dir / "ffn_w2_weight.bin")
        save_tensor(ffn_b2, out_dir / "ffn_w2_bias.bin")

    if dump_logits:
        save_tensor(lm_head_w, out_dir / "lm_head_weight.bin")
        save_tensor(lm_head_b, out_dir / "lm_head_bias.bin")

    save_tensor(tok_embed, out_dir / "tok_embed.bin")
    save_tensor(pos_embed, out_dir / "pos_embed.bin")
    save_tensor(tok_plus_pos, out_dir / "tok_plus_pos.bin")
    if dump_ln1:
        save_tensor(ln1_out, out_dir / "ln1_out.bin")

    if dump_attn:
        save_tensor(q, out_dir / "q.bin")
        save_tensor(k, out_dir / "k.bin")
        save_tensor(v, out_dir / "v.bin")
        save_tensor(q_heads, out_dir / "q_heads.bin")
        save_tensor(k_heads, out_dir / "k_heads.bin")
        save_tensor(v_heads, out_dir / "v_heads.bin")
        save_tensor(scores, out_dir / "attn_scores.bin")
        save_tensor(scores_masked, out_dir / "attn_scores_masked.bin")
        save_tensor(attn_weights, out_dir / "attn_weights.bin")
        save_tensor(attn_out, out_dir / "attn_out.bin")
        save_tensor(attn_proj, out_dir / "attn_proj.bin")
        save_tensor(residual1, out_dir / "residual1.bin")

    if dump_ffn:
        save_tensor(ln2_out, out_dir / "ln2_out.bin")
        save_tensor(ffn_w1_out, out_dir / "ffn_w1_out.bin")
        save_tensor(ffn_gelu, out_dir / "ffn_gelu.bin")
        save_tensor(ffn_w2_out, out_dir / "ffn_w2_out.bin")
        save_tensor(output, out_dir / "output.bin")

    if dump_logits:
        save_tensor(logits, out_dir / "logits.bin")

    save_tensor(loss.view(1), out_dir / "loss.bin")
    save_tensor(target, out_dir / "target.bin")

    save_tensor(tok_weight.grad, out_dir / "tok_weight_grad.bin")
    save_tensor(pos_weight.grad, out_dir / "pos_weight_grad.bin")
    if dump_ln1:
        save_tensor(ln1_gamma.grad, out_dir / "ln1_gamma_grad.bin")
        save_tensor(ln1_beta.grad, out_dir / "ln1_beta_grad.bin")
    if dump_ffn:
        save_tensor(ln2_gamma.grad, out_dir / "ln2_gamma_grad.bin")
        save_tensor(ln2_beta.grad, out_dir / "ln2_beta_grad.bin")

    if dump_attn:
        save_tensor(wq.grad, out_dir / "wq_weight_grad.bin")
        save_tensor(wq_b.grad, out_dir / "wq_bias_grad.bin")
        save_tensor(wk.grad, out_dir / "wk_weight_grad.bin")
        save_tensor(wk_b.grad, out_dir / "wk_bias_grad.bin")
        save_tensor(wv.grad, out_dir / "wv_weight_grad.bin")
        save_tensor(wv_b.grad, out_dir / "wv_bias_grad.bin")
        save_tensor(wo.grad, out_dir / "wo_weight_grad.bin")
        save_tensor(wo_b.grad, out_dir / "wo_bias_grad.bin")

    if dump_ffn:
        save_tensor(ffn_w1.grad, out_dir / "ffn_w1_weight_grad.bin")
        save_tensor(ffn_b1.grad, out_dir / "ffn_w1_bias_grad.bin")
        save_tensor(ffn_w2.grad, out_dir / "ffn_w2_weight_grad.bin")
        save_tensor(ffn_b2.grad, out_dir / "ffn_w2_bias_grad.bin")

    if dump_logits:
        save_tensor(lm_head_w.grad, out_dir / "lm_head_weight_grad.bin")
        save_tensor(lm_head_b.grad, out_dir / "lm_head_bias_grad.bin")
        save_tensor(logits.grad, out_dir / "logits_grad.bin")

    print(f"Loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()
