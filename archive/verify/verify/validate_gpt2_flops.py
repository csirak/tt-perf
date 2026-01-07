#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Validate GPT-2 FLOPs calculation against PyTorch CPU reference.
# Config must match C++ benchmark: batch=32, seq=256, dim=512, heads=8, ffn_mult=4, layers=6

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

# Config - must match bench_gpt2_trace.cpp
BATCH = 32
SEQ = 256
DIM = 512
HEADS = 8
FFN_MULT = 4
VOCAB = 256
LAYERS = 6
HEAD_DIM = DIM // HEADS
FFN_DIM = DIM * FFN_MULT

def count_matmul_flops(M, K, N):
    """2*M*K*N for matmul"""
    return 2 * M * K * N

def print_flops_table():
    """Print expected FLOPs breakdown per layer"""
    M = BATCH * SEQ  # 8192

    print("=" * 70)
    print("GPT-2 FLOPs Breakdown")
    print(f"Config: batch={BATCH}, seq={SEQ}, dim={DIM}, heads={HEADS}, ffn_mult={FFN_MULT}")
    print(f"M = batch * seq = {M}")
    print("=" * 70)

    forward_flops = 0
    backward_flops = 0

    # Forward pass per layer
    print("\n### Forward Pass (per layer)")
    print(f"{'Operation':<25} {'Shape (M,K,N)':<20} {'FLOPs':>15}")
    print("-" * 60)

    # QKV projections (3x)
    qkv_flops = count_matmul_flops(M, DIM, DIM)
    print(f"{'Q projection':<25} {f'{M}x{DIM}x{DIM}':<20} {qkv_flops:>15,}")
    print(f"{'K projection':<25} {f'{M}x{DIM}x{DIM}':<20} {qkv_flops:>15,}")
    print(f"{'V projection':<25} {f'{M}x{DIM}x{DIM}':<20} {qkv_flops:>15,}")
    forward_flops += 3 * qkv_flops

    # Attention scores: [B,H,S,D/H] @ [B,H,D/H,S] -> [B,H,S,S]
    # Per head: S x D/H x S = 256 x 64 x 256
    # All heads: B*H * S * D/H * S = 32*8 * 256 * 64 * 256 / 2 (since we count 2*MKN)
    attn_scores_flops = count_matmul_flops(BATCH * HEADS * SEQ, HEAD_DIM, SEQ)
    print(f"{'Attention scores':<25} {f'{BATCH*HEADS}x{SEQ}x{HEAD_DIM}x{SEQ}':<20} {attn_scores_flops:>15,}")
    forward_flops += attn_scores_flops

    # Attention output: [B,H,S,S] @ [B,H,S,D/H] -> [B,H,S,D/H]
    attn_out_flops = count_matmul_flops(BATCH * HEADS * SEQ, SEQ, HEAD_DIM)
    print(f"{'Attention output':<25} {f'{BATCH*HEADS}x{SEQ}x{SEQ}x{HEAD_DIM}':<20} {attn_out_flops:>15,}")
    forward_flops += attn_out_flops

    # Output projection
    out_proj_flops = count_matmul_flops(M, DIM, DIM)
    print(f"{'Output projection':<25} {f'{M}x{DIM}x{DIM}':<20} {out_proj_flops:>15,}")
    forward_flops += out_proj_flops

    # FFN layer 1: D -> FFN_DIM
    ffn1_flops = count_matmul_flops(M, DIM, FFN_DIM)
    print(f"{'FFN up (w1)':<25} {f'{M}x{DIM}x{FFN_DIM}':<20} {ffn1_flops:>15,}")
    forward_flops += ffn1_flops

    # FFN layer 2: FFN_DIM -> D
    ffn2_flops = count_matmul_flops(M, FFN_DIM, DIM)
    print(f"{'FFN down (w2)':<25} {f'{M}x{FFN_DIM}x{DIM}':<20} {ffn2_flops:>15,}")
    forward_flops += ffn2_flops

    print("-" * 60)
    print(f"{'Forward per layer':<25} {'':<20} {forward_flops:>15,}")
    print(f"{'Forward per layer (GFLOPs)':<25} {'':<20} {forward_flops/1e9:>15.2f}")

    # Backward pass per layer (roughly 2x forward for matmuls)
    # d_weight = d_out.T @ x (same shape as forward)
    # d_input = d_out @ weight (same shape as forward)
    backward_flops = 2 * forward_flops  # Approximation

    print("\n### Backward Pass (per layer)")
    print(f"Backward ≈ 2x Forward = {backward_flops:,} FLOPs")
    print(f"Backward per layer (GFLOPs): {backward_flops/1e9:.2f}")

    # Total
    total_per_layer = forward_flops + backward_flops
    total_all_layers = total_per_layer * LAYERS

    print("\n### Total FLOPs")
    print(f"Per layer (forward + backward): {total_per_layer:,} = {total_per_layer/1e9:.2f} GFLOPs")
    print(f"All {LAYERS} layers: {total_all_layers:,} = {total_all_layers/1e9:.2f} GFLOPs")

    # Expected time
    tflops_sustained = 250  # Estimated sustained TFLOPS for Wormhole
    expected_time = total_all_layers / (tflops_sustained * 1e12)
    print(f"\nExpected time at {tflops_sustained} TFLOPS: {expected_time*1000:.1f} ms")

    return total_all_layers


class SimpleTransformerLayer(nn.Module):
    """Simplified transformer layer matching our C++ implementation"""
    def __init__(self, dim, heads, ffn_mult):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # QKV projections
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        # FFN
        self.w1 = nn.Linear(dim, dim * ffn_mult, bias=False)
        self.w2 = nn.Linear(dim * ffn_mult, dim, bias=False)

        # Layer norms
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, S, D = x.shape

        # Self-attention
        residual = x
        x = self.ln1(x)

        q = self.wq(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Causal mask
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.wo(out)
        x = residual + out

        # FFN
        residual = x
        x = self.ln2(x)
        x = self.w1(x)
        x = F.gelu(x)
        x = self.w2(x)
        x = residual + x

        return x


class SimpleGPT2(nn.Module):
    def __init__(self, vocab, dim, heads, ffn_mult, layers):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(dim, heads, ffn_mult) for _ in range(layers)
        ])
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.head(x)


def run_pytorch_benchmark():
    """Run PyTorch CPU benchmark to measure actual time"""
    print("\n" + "=" * 70)
    print("PyTorch CPU Benchmark")
    print("=" * 70)

    device = torch.device('cpu')
    model = SimpleGPT2(VOCAB, DIM, HEADS, FFN_MULT, LAYERS).to(device)
    model.train()

    # Random tokens
    tokens = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)
    target = torch.randint(0, VOCAB, (BATCH, SEQ), device=device)

    # Warmup
    for _ in range(3):
        logits = model(tokens)
        loss = F.cross_entropy(logits.view(-1, VOCAB), target.view(-1))
        loss.backward()
        model.zero_grad()

    # Timed runs
    n_iter = 5
    start = time.time()
    for _ in range(n_iter):
        logits = model(tokens)
        loss = F.cross_entropy(logits.view(-1, VOCAB), target.view(-1))
        loss.backward()
        model.zero_grad()
    end = time.time()

    avg_time = (end - start) / n_iter * 1000  # ms
    print(f"Average time per step: {avg_time:.1f} ms")
    print(f"(Note: This is CPU - Wormhole should be much faster)")

    return avg_time


if __name__ == "__main__":
    total_flops = print_flops_table()
    run_pytorch_benchmark()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total FLOPs per training step: {total_flops/1e9:.2f} GFLOPs")
    print(f"Actual C++ benchmark reported: 411 ms")
    print(f"Implied TFLOPS: {total_flops / (0.411) / 1e12:.1f} TFLOPS")
