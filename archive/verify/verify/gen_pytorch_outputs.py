#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Generate PyTorch reference outputs for attention verification.
# Saves weights, input, and all intermediate outputs to binary files.
#
# Usage:
#   python3 verify/gen_pytorch_outputs.py

import torch
import torch.nn.functional as F
import math
from pathlib import Path
from tensor_io import save_tensor

# Configuration - must match C++ verification
BATCH = 32
SEQ = 64
DIM = 512
HEADS = 8
HEAD_DIM = DIM // HEADS
SEED = 42

OUTPUT_DIR = Path("verify/outputs/pytorch")


def create_causal_mask(seq_len: int, dtype=torch.bfloat16) -> torch.Tensor:
    """Create causal mask: upper triangular = -1e9 (not -inf for numerical comparison)."""
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype) * -1e9, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]


def main():
    print("=" * 70)
    print("Generating PyTorch Reference Outputs")
    print("=" * 70)
    print(f"Config: batch={BATCH}, seq={SEQ}, dim={DIM}, heads={HEADS}")
    print(f"Seed: {SEED}")
    print(f"Output dir: {OUTPUT_DIR}")

    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate weights (constant init like GPT-2 std=0.02)
    # Using constant for reproducibility - can verify exact match
    init_val = 0.02

    wq = torch.full((DIM, DIM), init_val, dtype=torch.bfloat16)
    wk = torch.full((DIM, DIM), init_val, dtype=torch.bfloat16)
    wv = torch.full((DIM, DIM), init_val, dtype=torch.bfloat16)
    wo = torch.full((DIM, DIM), init_val, dtype=torch.bfloat16)

    # Save weights
    print("\nSaving weights...")
    save_tensor(wq, OUTPUT_DIR / "wq.bin")
    save_tensor(wk, OUTPUT_DIR / "wk.bin")
    save_tensor(wv, OUTPUT_DIR / "wv.bin")
    save_tensor(wo, OUTPUT_DIR / "wo.bin")
    print(f"  wq, wk, wv, wo: [{DIM}, {DIM}]")

    # Generate input (small random values for stable attention)
    torch.manual_seed(SEED)  # Reset for reproducible input
    x = torch.randn(BATCH, SEQ, DIM, dtype=torch.bfloat16) * 0.1
    save_tensor(x, OUTPUT_DIR / "input.bin")
    print(f"  input: [{BATCH}, {SEQ}, {DIM}]")

    # Create causal mask
    mask = create_causal_mask(SEQ)
    save_tensor(mask, OUTPUT_DIR / "causal_mask.bin")
    print(f"  causal_mask: [1, 1, {SEQ}, {SEQ}]")

    # Forward pass with intermediate saves
    print("\nRunning forward pass...")

    # QKV projections: y = x @ W.T (Linear without bias)
    q_proj = F.linear(x, wq)  # [B, S, D]
    k_proj = F.linear(x, wk)
    v_proj = F.linear(x, wv)

    save_tensor(q_proj, OUTPUT_DIR / "q_proj.bin")
    save_tensor(k_proj, OUTPUT_DIR / "k_proj.bin")
    save_tensor(v_proj, OUTPUT_DIR / "v_proj.bin")
    print(f"  q_proj, k_proj, v_proj: [{BATCH}, {SEQ}, {DIM}]")

    # Reshape for multi-head: [B, S, D] -> [B, H, S, D/H]
    q = q_proj.view(BATCH, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
    k = k_proj.view(BATCH, SEQ, HEADS, HEAD_DIM).transpose(1, 2)
    v = v_proj.view(BATCH, SEQ, HEADS, HEAD_DIM).transpose(1, 2)

    save_tensor(q, OUTPUT_DIR / "q.bin")
    save_tensor(k, OUTPUT_DIR / "k.bin")
    save_tensor(v, OUTPUT_DIR / "v.bin")
    print(f"  q, k, v (reshaped): [{BATCH}, {HEADS}, {SEQ}, {HEAD_DIM}]")

    # Attention scores: Q @ K.T * scale
    scale = 1.0 / math.sqrt(HEAD_DIM)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    save_tensor(scores, OUTPUT_DIR / "scores.bin")
    print(f"  scores: [{BATCH}, {HEADS}, {SEQ}, {SEQ}]")

    # Apply causal mask
    scores_masked = scores + mask
    save_tensor(scores_masked, OUTPUT_DIR / "scores_masked.bin")
    print(f"  scores_masked: [{BATCH}, {HEADS}, {SEQ}, {SEQ}]")

    # Softmax
    attn_weights = F.softmax(scores_masked, dim=-1)
    save_tensor(attn_weights, OUTPUT_DIR / "attn_weights.bin")
    print(f"  attn_weights: [{BATCH}, {HEADS}, {SEQ}, {SEQ}]")

    # Attention output: weights @ V
    attn_out = torch.matmul(attn_weights, v)
    save_tensor(attn_out, OUTPUT_DIR / "attn_out.bin")
    print(f"  attn_out: [{BATCH}, {HEADS}, {SEQ}, {HEAD_DIM}]")

    # Merge heads: [B, H, S, D/H] -> [B, S, D]
    attn_merged = attn_out.transpose(1, 2).contiguous().view(BATCH, SEQ, DIM)
    save_tensor(attn_merged, OUTPUT_DIR / "attn_merged.bin")
    print(f"  attn_merged: [{BATCH}, {SEQ}, {DIM}]")

    # Output projection
    output = F.linear(attn_merged, wo)
    save_tensor(output, OUTPUT_DIR / "output.bin")
    print(f"  output: [{BATCH}, {SEQ}, {DIM}]")

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"  Input mean:  {x.float().mean().item():.6f}, std: {x.float().std().item():.6f}")
    print(f"  Output mean: {output.float().mean().item():.6f}, std: {output.float().std().item():.6f}")
    print(f"  Attn weights sum (should be ~1): {attn_weights[0, 0, 0, :].sum().item():.6f}")

    print(f"\nSaved {len(list(OUTPUT_DIR.glob('*.bin')))} files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
