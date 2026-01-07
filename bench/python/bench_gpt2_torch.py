#!/usr/bin/env python3
"""
PyTorch GPT-2 Training Baseline for loss comparison.

Architecture matches our C++ TracedTransformerWithEmbedding:
  - Token embedding [vocab, dim]
  - Positional embedding [seq, dim]
  - N x TransformerBlock (pre-norm: LN -> Attention -> Add -> LN -> FFN -> Add)
  - Output projection [dim, vocab]

Usage:
  python bench/python/bench_gpt2_torch.py
  BENCH_LAYERS=6 python bench/python/bench_gpt2_torch.py
"""

import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention matching our C++ implementation."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        H = self.heads

        q = self.wq(x).view(B, S, H, self.head_dim).transpose(1, 2)  # [B, H, S, D/H]
        k = self.wk(x).view(B, S, H, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, H, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]
        mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device, dtype=x.dtype), diagonal=1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, S, D/H]

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.wo(out)


class TransformerBlock(nn.Module):
    """GPT-2 style transformer block with pre-norm."""

    def __init__(self, dim: int, heads: int, ffn_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = CausalSelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim, eps=1e-5)
        self.w1 = nn.Linear(dim, dim * ffn_mult, bias=False)
        self.w2 = nn.Linear(dim * ffn_mult, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attn(self.ln1(x))
        # Pre-norm FFN with GELU and residual
        x = x + self.w2(F.gelu(self.w1(self.ln2(x))))
        return x


class GPT2(nn.Module):
    """GPT-2 model matching our C++ TracedTransformerWithEmbedding."""

    def __init__(self, vocab: int, dim: int, seq: int, heads: int,
                 ffn_mult: int, num_layers: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(seq, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, ffn_mult)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(dim, vocab, bias=False)

        # Initialize weights similar to our C++ (small init for stability)
        self._init_weights()

    def _init_weights(self):
        # Match C++ constant fill initialization for fair comparison
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0.01)  # C++ uses 0.01 for linear
            elif isinstance(module, nn.Embedding):
                nn.init.constant_(module.weight, 0.02)  # C++ uses 0.02 for embeddings
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, S = token_ids.shape
        pos = torch.arange(S, device=token_ids.device).unsqueeze(0).expand(B, -1)

        x = self.tok_emb(token_ids) + self.pos_emb(pos)

        for layer in self.layers:
            x = layer(x)

        return self.output_proj(x)


def main():
    # Config from environment (matching C++ defaults)
    # All dims must be multiples of 32, dim/heads must be multiple of 32
    batch = int(os.getenv("BENCH_BATCH", "32"))
    seq = int(os.getenv("BENCH_SEQ", "256"))
    dim = int(os.getenv("BENCH_DIM", "512"))      # 512 = 8 heads * 64 head_dim
    heads = int(os.getenv("BENCH_HEADS", "8"))
    ffn_mult = int(os.getenv("BENCH_FFN_MULT", "4"))
    vocab = int(os.getenv("BENCH_VOCAB", "256"))
    num_layers = int(os.getenv("BENCH_LAYERS", "6"))
    lr = float(os.getenv("BENCH_LR", "0.01"))
    n_warmup = int(os.getenv("N_WARMUP", "3"))
    n_iters = int(os.getenv("N_ITERS", "10"))

    print(f"# PyTorch GPT-2 Training Baseline")
    print(f"# Config: batch={batch}, seq={seq}, dim={dim}, heads={heads}, ffn_mult={ffn_mult}, vocab={vocab}, layers={num_layers}")
    print(f"# LR: {lr}, Warmup: {n_warmup}, Iterations: {n_iters}")

    # Use bfloat16 to match TT hardware
    dtype = torch.bfloat16
    device = "cpu"  # CPU for fair comparison (no GPU)

    # Create model
    model = GPT2(vocab, dim, seq, heads, ffn_mult, num_layers).to(dtype).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Create random input (same seed as C++)
    torch.manual_seed(42)
    token_ids = torch.randint(0, vocab, (batch, seq), device=device)
    target = torch.zeros(batch, seq, vocab, dtype=dtype, device=device)

    # Warmup
    for _ in range(n_warmup):
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = F.mse_loss(logits, target)
        loss.backward()
        optimizer.step()

    # Timed runs with loss tracking
    print(f"\n# Loss trajectory (step, loss, time_ms)")
    total_time = 0.0
    for i in range(n_iters):
        start = time.perf_counter()
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = F.mse_loss(logits, target)
        loss.backward()
        optimizer.step()
        end = time.perf_counter()

        step_ms = (end - start) * 1000
        total_time += step_ms
        print(f"{i},{loss.item():.6f},{step_ms:.3f}")

    avg_ms = total_time / n_iters
    print(f"\n# Average time: {avg_ms:.3f} ms/step")


if __name__ == "__main__":
    main()
