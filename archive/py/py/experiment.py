import argparse
import csv
import os
import math
import sys
import time

import torch
import ttnn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from grad.data import ModularDivisionDataset
from grad.engine import Tensor, resolve_tt_dtype, resolve_torch_dtype
from grad.nn import Embedding, TransformerBlock, Linear
from grad.optim import Adam


def _build_loss_cache(batch: int, seq: int, classes: int, answer_pos: int, device, dtype):
    row_indices = torch.arange(batch, dtype=torch.int64) * seq + answer_pos
    row_mask = torch.zeros((batch * seq, classes), dtype=resolve_torch_dtype(dtype))
    row_mask[row_indices, :] = 1.0
    row_mask_tt = ttnn.from_torch(
        row_mask,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=resolve_tt_dtype(dtype),
    )
    row_mask_1d_torch = torch.zeros((batch * seq,), dtype=resolve_torch_dtype(dtype))
    row_mask_1d_torch[row_indices] = 1.0
    row_mask_1d = ttnn.from_torch(
        row_mask_1d_torch,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=resolve_tt_dtype(dtype),
    )
    index_buf = torch.zeros((batch * seq,), dtype=torch.int64)
    identity = torch.eye(classes, dtype=resolve_torch_dtype(dtype))
    one_hot_weight_tt = ttnn.from_torch(
        identity,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=resolve_tt_dtype(dtype),
    )
    return row_indices, row_mask_tt, row_mask_1d, index_buf, one_hot_weight_tt


def last_token_cross_entropy(
    logits: Tensor,
    targets: torch.Tensor,
    answer_pos: int,
    *,
    row_indices: torch.Tensor,
    row_mask_tt: ttnn.Tensor,
    row_mask_1d: ttnn.Tensor,
    index_buf: torch.Tensor,
    one_hot_weight_tt: ttnn.Tensor,
):
    if not isinstance(logits, Tensor):
        raise ValueError("logits must be a Tensor")
    if logits.compute_kernel_config is None:
        raise ValueError("compute_kernel_config must be set")

    batch, seq, classes = logits.tt.shape
    if targets.shape != (batch, 1):
        raise ValueError("targets must have shape (B, 1)")
    if answer_pos < 0 or answer_pos >= seq:
        raise ValueError("answer_pos out of range")

    flat_logits = ttnn.reshape(logits.tt, (batch * seq, classes))
    index_buf.zero_()
    index_buf[row_indices] = targets[:, 0].to(torch.int64)
    indices_tt = ttnn.from_torch(
        index_buf,
        device=logits.device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    one_hot_tt = ttnn.embedding(
        indices_tt,
        weight=one_hot_weight_tt,
        dtype=resolve_tt_dtype(logits.tt.dtype),
        layout=ttnn.TILE_LAYOUT,
    )

    softmax_tt = ttnn.softmax(flat_logits, dim=-1, compute_kernel_config=logits.compute_kernel_config)
    log_probs_tt = ttnn.log(softmax_tt)
    nll_tt = ttnn.sum(ttnn.mul(one_hot_tt, log_probs_tt), dim=-1, keepdim=False)
    nll_tt = ttnn.mul(nll_tt, ttnn.full_like(nll_tt, fill_value=-1.0))
    nll_tt = ttnn.mul(nll_tt, row_mask_1d)
    total = ttnn.sum(nll_tt, dim=0, keepdim=False)
    scale_tt = ttnn.full_like(total, fill_value=1.0 / float(batch))
    loss_tt = ttnn.mul(total, scale_tt)

    out = Tensor(
        loss_tt,
        (logits,),
        "last_token_cross_entropy",
        logits.requires_grad,
        compute_kernel_config=logits.compute_kernel_config,
    )

    def _backward():
        if not logits.requires_grad:
            return
        grad_scale = ttnn.to_torch(out.grad).item()
        grad_scale /= float(batch)
        scale_tt = ttnn.full_like(softmax_tt, fill_value=grad_scale)
        grad_logits = ttnn.sub(softmax_tt, one_hot_tt)
        grad_logits = ttnn.mul(grad_logits, row_mask_tt)
        grad_logits = ttnn.mul(grad_logits, scale_tt)
        grad_logits = ttnn.reshape(grad_logits, (batch, seq, classes))
        logits._add_grad(grad_logits)

    out._backward = _backward
    return out


class TinyTransformer:
    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        dim: int,
        heads: int,
        ffn_mult: int,
        device,
        dtype=None,
        batch_size: int = None,
    ):
        self.tok = Embedding(vocab_size, dim, dtype=dtype, device=device)
        self.pos = Embedding(seq_len, dim, dtype=dtype, device=device)
        self.blocks = [
            TransformerBlock(dim, heads=heads, ffn_mult=ffn_mult, dtype=dtype, device=device),
            TransformerBlock(dim, heads=heads, ffn_mult=ffn_mult, dtype=dtype, device=device),
        ]
        self.out = Linear(dim, vocab_size, dtype=dtype, device=device)
        self._pos_idx = None
        if batch_size is not None:
            pos_idx = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
            pos_idx_tt = ttnn.from_torch(
                pos_idx,
                device=device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
            )
            pos_tensor = Tensor(pos_idx_tt, device=device)
            pos_tensor._torch_cache = pos_idx
            self._pos_idx = pos_tensor

    def parameters(self):
        params = []
        params.extend(self.tok.parameters())
        params.extend(self.pos.parameters())
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.out.parameters())
        return params

    def __call__(self, tokens):
        if isinstance(tokens, Tensor):
            batch, seq = tokens.tt.shape
        else:
            batch, seq = tokens.shape
        if self._pos_idx is not None:
            if self._pos_idx.tt.shape[0] != batch or self._pos_idx.tt.shape[1] != seq:
                raise ValueError("batch/seq mismatch with cached positional indices")
            pos_idx = self._pos_idx
        else:
            pos_idx = torch.arange(seq, dtype=torch.int64).unsqueeze(0).repeat(batch, 1)
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


def _slice_batch(tokens_tt, tokens_host, targets_host, start: int, batch_size: int):
    end = start + batch_size
    tokens_tt_batch = tokens_tt[start:end]
    tokens_host_batch = tokens_host[start:end]
    targets_batch = targets_host[start:end]
    tokens = Tensor(tokens_tt_batch, device=tokens_tt.device())
    tokens._torch_cache = tokens_host_batch
    return tokens, targets_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--csv", type=str, default="experiment.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--pad-to", type=int, default=32)
    parser.add_argument("--profile-read-every", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.csv:
        open(args.csv, "w").close()
    dataset = ModularDivisionDataset(p=97, train_frac=0.5, pad_to=args.pad_to, seed=args.seed)
    train_tokens_host, train_targets_host = dataset.get_split_tensors("train")
    val_tokens_host, val_targets_host = dataset.get_split_tensors("val")
    train_tokens_host, train_targets_host = _pad_to_batch(train_tokens_host, train_targets_host, args.batch_size)
    val_tokens_host, val_targets_host = _pad_to_batch(val_tokens_host, val_targets_host, args.batch_size)

    profile_env = os.environ.get("TT_METAL_DEVICE_PROFILER", "").lower() in ("1", "true", "yes")
    profile_read_every = args.profile_read_every or (1 if profile_env else 0)

    device = None
    device = ttnn.open_device(device_id=args.device_id)
    try:
        train_tokens_tt = ttnn.from_torch(
            train_tokens_host,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )
        val_tokens_tt = ttnn.from_torch(
            val_tokens_host,
            device=device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )
        model = TinyTransformer(
            vocab_size=dataset.vocab_size,
            seq_len=dataset.seq_len,
            dim=args.dim,
            heads=args.heads,
            ffn_mult=args.ffn_mult,
            device=device,
            batch_size=args.batch_size,
        )
        opt = Adam(model.parameters(), lr=args.lr)
        row_indices, row_mask_tt, row_mask_1d, index_buf, one_hot_weight_tt = _build_loss_cache(
            args.batch_size,
            dataset.seq_len,
            dataset.vocab_size,
            dataset.answer_pos,
            device,
            model.out.weight.tt.dtype,
        )

        train_pos = 0
        val_pos = 0
        start_time = time.perf_counter()
        last_log_time = start_time
        for step in range(1, args.steps + 1):
            tokens, targets = _slice_batch(
                train_tokens_tt,
                train_tokens_host,
                train_targets_host,
                train_pos,
                args.batch_size,
            )
            train_pos = (train_pos + args.batch_size) % train_tokens_host.shape[0]
            logits = model(tokens)
            loss = last_token_cross_entropy(
                logits,
                targets,
                dataset.answer_pos,
                row_indices=row_indices,
                row_mask_tt=row_mask_tt,
                row_mask_1d=row_mask_1d,
                index_buf=index_buf,
                one_hot_weight_tt=one_hot_weight_tt,
            )
            loss.backward()
            opt.step()
            opt.zero_grad()
            if profile_read_every > 0 and (step % profile_read_every) == 0:
                ttnn.ReadDeviceProfiler(device)

            if step % args.log_every == 0:
                train_loss = loss.to_torch().float().item()
                tokens_val, targets_val = _slice_batch(
                    val_tokens_tt,
                    val_tokens_host,
                    val_targets_host,
                    val_pos,
                    args.batch_size,
                )
                val_pos = (val_pos + args.batch_size) % val_tokens_host.shape[0]
                with torch.no_grad():
                    logits_val = model(tokens_val)
                    val_loss = last_token_cross_entropy(
                        logits_val,
                        targets_val,
                        dataset.answer_pos,
                        row_indices=row_indices,
                        row_mask_tt=row_mask_tt,
                        row_mask_1d=row_mask_1d,
                        index_buf=index_buf,
                        one_hot_weight_tt=one_hot_weight_tt,
                    )
                    val_loss = val_loss.to_torch().float().item()
                now = time.perf_counter()
                interval_s = now - last_log_time
                total_s = now - start_time
                last_log_time = now
                _append_csv(args.csv, step, train_loss, val_loss, interval_s, total_s)
    finally:
        if device is not None and profile_env:
            ttnn.ReadDeviceProfiler(device)
        if device is not None:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
