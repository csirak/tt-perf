#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch training loop that mirrors the TTNN grok model + hyperparams.

import argparse
import math
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_kv_yaml(path: Path) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        cfg[key.strip()] = val.strip()
    return cfg


def get_int(cfg: Dict[str, str], key: str, default: int) -> int:
    return int(cfg.get(key, default))


def get_float(cfg: Dict[str, str], key: str, default: float) -> float:
    return float(cfg.get(key, default))


def get_str(cfg: Dict[str, str], key: str, default: str) -> str:
    return str(cfg.get(key, default))


def next_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def mod_div(x: int, y: int, p: int) -> int:
    return (x * pow(y, p - 2, p)) % p


class ModularDivisionDataset:
    def __init__(self, *, p: int, train_frac: float, pad_to: int, seed: int):
        if p <= 2:
            raise ValueError("p must be > 2")
        if not (0.0 < train_frac < 1.0):
            raise ValueError("train_frac must be in (0, 1)")
        if pad_to < 4 or (pad_to % 32) != 0:
            raise ValueError("pad_to must be >= 4 and a multiple of 32")
        self.p = int(p)
        self.train_frac = float(train_frac)
        self.pad_to = int(pad_to)
        self.seq_len = int(pad_to)
        self.div_token = self.p
        self.eq_token = self.p + 1
        self.base_vocab = self.p + 2
        self.vocab_size = next_multiple(self.base_vocab, 32)
        self.answer_pos = 3

        xs = []
        ys = []
        zs = []
        for x in range(self.p):
            for y in range(1, self.p):
                xs.append(x)
                ys.append(y)
                zs.append(mod_div(x, y, self.p))

        total = len(xs)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(total, generator=g)
        split = int(math.floor(total * self.train_frac))
        self.train_idx = perm[:split]
        self.val_idx = perm[split:]

        self._x_all = torch.tensor(xs, dtype=torch.int64)
        self._y_all = torch.tensor(ys, dtype=torch.int64)
        self._z_all = torch.tensor(zs, dtype=torch.int64)

        self._rng = torch.Generator().manual_seed(seed)

    def sample_batch(self, batch_size: int, train_split: bool) -> tuple[torch.Tensor, torch.Tensor]:
        idx_pool = self.train_idx if train_split else self.val_idx
        if idx_pool.numel() == 0:
            raise RuntimeError("empty split")
        idx = torch.randint(0, idx_pool.numel(), (batch_size,), generator=self._rng)
        sel = idx_pool[idx]
        x = self._x_all[sel]
        y = self._y_all[sel]
        z = self._z_all[sel]

        tokens = torch.zeros((batch_size, self.seq_len), dtype=torch.int64)
        tokens[:, 0] = x
        tokens[:, 1] = self.div_token
        tokens[:, 2] = y
        tokens[:, 3] = self.eq_token
        targets = z
        return tokens, targets


class TorchBlockLN(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_mult: int, ln_eps: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.ln1 = nn.LayerNorm(dim, eps=ln_eps)
        self.wq = nn.Linear(dim, dim, bias=True)
        self.wk = nn.Linear(dim, dim, bias=True)
        self.wv = nn.Linear(dim, dim, bias=True)
        self.wo = nn.Linear(dim, dim, bias=True)

        self.ln2 = nn.LayerNorm(dim, eps=ln_eps)
        self.ffn1 = nn.Linear(dim, dim * ffn_mult, bias=True)
        self.ffn2 = nn.Linear(dim * ffn_mult, dim, bias=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        batch, seq, dim = x.shape
        h = F.layer_norm(x.float(), (dim,), self.ln1.weight.float(), self.ln1.bias.float(), self.ln1.eps).to(dtype)

        q = (self.wq(h).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)).to(dtype)
        k = (self.wk(h).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)).to(dtype)
        v = (self.wv(h).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)).to(dtype)

        scores = torch.matmul(q, k.transpose(-1, -2)) * torch.tensor(self.scale, dtype=dtype)
        scores_masked = (scores + mask).to(dtype)
        scores_max = scores_masked.max(dim=-1, keepdim=True).values.to(dtype)
        scores_centered = (scores_masked - scores_max).to(dtype)
        attn = torch.softmax(scores_centered, dim=-1).to(dtype)
        attn_out = torch.matmul(attn, v).to(dtype)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, dim)
        attn_out = self.wo(attn_out).to(dtype)
        x = (x + attn_out).to(dtype)

        h2 = F.layer_norm(x.float(), (dim,), self.ln2.weight.float(), self.ln2.bias.float(), self.ln2.eps).to(dtype)
        ffn = self.ffn2(F.gelu(self.ffn1(h2), approximate="none")).to(dtype)
        return (x + ffn).to(dtype)


class TorchGrokModel(nn.Module):
    def __init__(self, *, vocab_size: int, seq_len: int, dim: int, heads: int, ffn_mult: int, layers: int, ln_eps: float):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([TorchBlockLN(dim, heads, ffn_mult, ln_eps) for _ in range(layers)])
        self.ln_final = nn.LayerNorm(dim, eps=ln_eps)
        self.out = nn.Linear(dim, vocab_size, bias=True)

    def forward(self, tokens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        batch, seq = tokens.shape
        pos_idx = torch.arange(seq, device=tokens.device).unsqueeze(0).expand(batch, -1)
        x = (self.tok(tokens) + self.pos(pos_idx)).to(dtype)
        mask = torch.triu(
            torch.full((1, 1, seq, seq), -1e9, dtype=dtype, device=x.device),
            diagonal=1,
        )
        for block in self.blocks:
            x = block(x, mask, dtype)
        x = F.layer_norm(x.float(), (x.shape[-1],), self.ln_final.weight.float(), self.ln_final.bias.float(),
                         self.ln_final.eps).to(dtype)
        return self.out(x).to(dtype)


def init_weights_like_ttnn(model: TorchGrokModel, dim: int):
    # Embeddings: N(0,1)
    nn.init.normal_(model.tok.weight, mean=0.0, std=1.0)
    nn.init.normal_(model.pos.weight, mean=0.0, std=1.0)

    # Linear weights: N(0, sqrt(1/d))
    std = math.sqrt(1.0 / dim)
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, mean=0.0, std=std)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        if isinstance(mod, nn.LayerNorm):
            nn.init.ones_(mod.weight)
            nn.init.zeros_(mod.bias)


def write_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("step\ttrain_loss\tval_loss\tinterval_s\ttotal_s\tavg_step_s\n")


def append_row(path: Path, step: int, train_loss: float, val_loss: float, interval_s: float, total_s: float):
    avg_step_s = total_s / step if step > 0 else 0.0
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{step}\t{train_loss}\t{val_loss}\t{interval_s:.6f}\t{total_s:.6f}\t{avg_step_s:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch grok training mirror")
    parser.add_argument("--config", type=str, default="experiments/grok/adam_tf08_1k.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = load_kv_yaml(Path(args.config))
    p = get_int(cfg, "p", 127)
    train_frac = get_float(cfg, "train_frac", 0.8)
    pad_to = get_int(cfg, "pad_to", 32)
    batch_size = get_int(cfg, "batch_size", 256)
    steps = get_int(cfg, "steps", 1000)
    log_every = get_int(cfg, "log_every", 100)
    eval_every = get_int(cfg, "eval_every", 100)
    seed = get_int(cfg, "seed", 0)

    lr = get_float(cfg, "lr", 1e-3)
    beta1 = get_float(cfg, "beta1", 0.9)
    beta2 = get_float(cfg, "beta2", 0.999)
    eps = get_float(cfg, "eps", 1e-6)
    wd = get_float(cfg, "weight_decay", 0.01)

    dim = get_int(cfg, "dim", 128)
    heads = get_int(cfg, "heads", 4)
    ffn_mult = get_int(cfg, "ffn_mult", 4)
    layers = get_int(cfg, "layers", 2)
    ln_eps = get_float(cfg, "ln_eps", 1e-5)

    out_path = Path(get_str(cfg, "csv", "/tmp/torch_grok_results.tsv"))
    if out_path.exists():
        out_path.unlink()
    write_header(out_path)

    torch.manual_seed(seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16

    dataset = ModularDivisionDataset(p=p, train_frac=train_frac, pad_to=pad_to, seed=seed)
    model = TorchGrokModel(vocab_size=dataset.vocab_size, seq_len=dataset.seq_len,
                           dim=dim, heads=heads, ffn_mult=ffn_mult, layers=layers, ln_eps=ln_eps).to(device)
    init_weights_like_ttnn(model, dim)
    model = model.to(dtype)

    # AdamW with decoupled WD; exclude biases + LN params
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optim = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
    )

    start = time.perf_counter()
    last = start
    for step in range(1, steps + 1):
        tokens, targets = dataset.sample_batch(batch_size, True)
        tokens = tokens.to(device)
        targets = targets.to(device)

        logits = model(tokens, dtype)
        logits_last = logits[:, dataset.answer_pos, :]
        log_probs = torch.log_softmax(logits_last.float(), dim=-1)
        loss = -log_probs[torch.arange(batch_size), targets].mean()
        loss.backward()
        optim.step()
        optim.zero_grad()

        if step % log_every == 0:
            train_loss = float(loss.detach().item())
            val_loss = train_loss
            if eval_every > 0 and step % eval_every == 0:
                with torch.no_grad():
                    vt, vtgt = dataset.sample_batch(batch_size, False)
                    vt = vt.to(device)
                    vtgt = vtgt.to(device)
                    v_logits = model(vt, dtype)
                    v_last = v_logits[:, dataset.answer_pos, :]
                    v_log_probs = torch.log_softmax(v_last.float(), dim=-1)
                    val_loss = float((-v_log_probs[torch.arange(batch_size), vtgt].mean()).item())
            now = time.perf_counter()
            interval_s = now - last
            total_s = now - start
            last = now
            append_row(out_path, step, train_loss, val_loss, interval_s, total_s)
            print(f"step {step} train_loss {train_loss:.6f} val_loss {val_loss:.6f} interval_s {interval_s:.2f} total_s {total_s:.2f}")


if __name__ == "__main__":
    main()
