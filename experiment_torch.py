import argparse
import csv
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def _next_multiple(value: int, multiple: int = 32) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _mod_div(x: int, y: int, p: int) -> int:
    return (x * pow(y, p - 2, p)) % p


class ModularDivisionDataset:
    def __init__(self, *, p: int = 97, train_frac: float = 0.5, pad_to: int = 32, seed: int = 0):
        if p <= 2:
            raise ValueError("p must be > 2")
        if not (0.0 < train_frac < 1.0):
            raise ValueError("train_frac must be in (0, 1)")
        if pad_to < 4 or (pad_to % 32) != 0:
            raise ValueError("pad_to must be >= 4 and a multiple of 32")
        self.p = int(p)
        self.div_token = self.p
        self.eq_token = self.p + 1
        self.base_vocab_size = self.p + 2
        self.vocab_size = _next_multiple(self.base_vocab_size, 32)
        self.seq_len = int(pad_to)
        self.answer_pos = 3

        xs = []
        ys = []
        zs = []
        for x in range(self.p):
            for y in range(1, self.p):
                xs.append(x)
                ys.append(y)
                zs.append(_mod_div(x, y, self.p))
        self._x_all = torch.tensor(xs, dtype=torch.int64)
        self._y_all = torch.tensor(ys, dtype=torch.int64)
        self._z_all = torch.tensor(zs, dtype=torch.int64)

        total = self._x_all.shape[0]
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(total, generator=gen)
        split = int(total * train_frac)
        self.train_idx = perm[:split]
        self.val_idx = perm[split:]

    def get_split_tensors(self, split: str):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        idx = self.train_idx if split == "train" else self.val_idx
        x = self._x_all[idx]
        y = self._y_all[idx]
        z = self._z_all[idx]
        div_tok = torch.full_like(x, self.div_token)
        eq_tok = torch.full_like(x, self.eq_token)
        tokens = torch.zeros((idx.numel(), self.seq_len), dtype=torch.int64)
        tokens[:, 0] = x
        tokens[:, 1] = div_tok
        tokens[:, 2] = y
        tokens[:, 3] = eq_tok
        targets = z.unsqueeze(1)
        return tokens, targets


class TorchBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_mult: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
        self.ffn1 = nn.Linear(dim, dim * ffn_mult, bias=False)
        self.ffn2 = nn.Linear(dim * ffn_mult, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        q = self.q(x).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq, seq, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq, dim)
        out = self.o(out)

        y = torch.tanh(x + out)
        ffn = self.ffn2(F.relu(self.ffn1(y)))
        return torch.tanh(y + ffn)


class TorchTinyTransformer(nn.Module):
    def __init__(self, *, vocab_size: int, seq_len: int, dim: int, heads: int, ffn_mult: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([TorchBlock(dim, heads, ffn_mult), TorchBlock(dim, heads, ffn_mult)])
        self.out = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq = tokens.shape
        pos_idx = torch.arange(seq, device=tokens.device).unsqueeze(0).repeat(batch, 1)
        x = self.tok(tokens) + self.pos(pos_idx)
        for block in self.blocks:
            x = block(x)
        return self.out(x)


def _append_csv(path: str, step: int, train_loss: float, val_loss: float, interval_s: float, total_s: float):
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "train_loss", "val_loss", "interval_s", "total_s"])
        writer.writerow([step, train_loss, val_loss, interval_s, total_s])


def _pad_to_batch(tokens: torch.Tensor, targets: torch.Tensor, batch_size: int):
    total = tokens.shape[0]
    pad = (-total) % batch_size
    if pad == 0:
        return tokens, targets
    return torch.cat([tokens, tokens[:pad]], dim=0), torch.cat([targets, targets[:pad]], dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--csv", type=str, default="experiment_torch.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--pad-to", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.csv:
        open(args.csv, "w").close()
    dataset = ModularDivisionDataset(p=97, train_frac=0.8, pad_to=args.pad_to, seed=args.seed)
    train_tokens, train_targets = dataset.get_split_tensors("train")
    val_tokens, val_targets = dataset.get_split_tensors("val")
    train_tokens, train_targets = _pad_to_batch(train_tokens, train_targets, args.batch_size)
    val_tokens, val_targets = _pad_to_batch(val_tokens, val_targets, args.batch_size)

    device = torch.device(args.device)
    model = TorchTinyTransformer(
        vocab_size=dataset.vocab_size,
        seq_len=dataset.seq_len,
        dim=args.dim,
        heads=args.heads,
        ffn_mult=args.ffn_mult,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_time = time.perf_counter()
    last_log_time = start_time
    train_pos = 0
    val_pos = 0
    for step in range(1, args.steps + 1):
        tokens = train_tokens[train_pos : train_pos + args.batch_size].to(device)
        targets = train_targets[train_pos : train_pos + args.batch_size].to(device)
        train_pos = (train_pos + args.batch_size) % train_tokens.shape[0]
        logits = model(tokens)
        loss = F.cross_entropy(logits[:, dataset.answer_pos, :], targets[:, 0])
        loss.backward()
        opt.step()
        opt.zero_grad()

        if step % args.log_every == 0:
            train_loss = loss.detach().item()
            with torch.no_grad():
                tokens_val = val_tokens[val_pos : val_pos + args.batch_size].to(device)
                targets_val = val_targets[val_pos : val_pos + args.batch_size].to(device)
                val_pos = (val_pos + args.batch_size) % val_tokens.shape[0]
                logits_val = model(tokens_val)
                val_loss = F.cross_entropy(logits_val[:, dataset.answer_pos, :], targets_val[:, 0]).item()
            now = time.perf_counter()
            interval_s = now - last_log_time
            total_s = now - start_time
            last_log_time = now
            _append_csv(args.csv, step, train_loss, val_loss, interval_s, total_s)


if __name__ == "__main__":
    main()
