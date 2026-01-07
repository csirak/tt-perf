#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Forward Pass Numerical Verification: TTNN vs PyTorch (BF16)
# Tests attention block with identical weights to measure numerical error.
#
# Usage:
#   cd ~/ttnn-perf && ./scripts/run.sh python3 verify/verify_forward.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
import math

# Configuration - smaller for quick iteration
BATCH = 32
SEQ = 64
DIM = 256
HEADS = 8
HEAD_DIM = DIM // HEADS
SEED = 42

torch.manual_seed(SEED)


def compare_tensors(name: str, ttnn_out: torch.Tensor, torch_out: torch.Tensor):
    """Compare TTNN output vs PyTorch reference. Both should be float for comparison."""
    a = ttnn_out.float().flatten()
    b = torch_out.float().flatten()

    diff = (a - b).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Relative error (avoid div by zero)
    denom = b.abs().clamp(min=1e-6)
    rel_err = (diff / denom).mean().item()

    # Also compute percentiles for distribution
    p50 = diff.median().item()
    p99 = diff.quantile(0.99).item()

    print(f"  {name:30s}: max={max_diff:.6f}, mean={mean_diff:.6f}, p99={p99:.6f}, rel={rel_err:.4f}")
    return {
        'name': name,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'p50': p50,
        'p99': p99,
        'rel_error': rel_err
    }


def create_causal_mask_torch(seq_len: int, dtype=torch.bfloat16) -> torch.Tensor:
    """Create causal mask for PyTorch: upper triangular = -inf."""
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype) * float('-inf'), diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]


def create_causal_mask_ttnn(seq_len: int, device) -> ttnn.Tensor:
    """Create causal mask for TTNN."""
    mask = torch.triu(torch.ones(1, 1, seq_len, seq_len, dtype=torch.bfloat16) * -1e9, diagonal=1)
    return ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


class PyTorchAttention(nn.Module):
    """Causal self-attention in PyTorch BF16."""

    def __init__(self, dim: int, heads: int, seq: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # QKV projections
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.register_buffer('mask', create_causal_mask_torch(seq))

    def forward(self, x: torch.Tensor) -> tuple:
        """Returns (output, intermediates) for comparison."""
        B, S, D = x.shape
        H = self.heads

        # QKV projections
        q_proj = self.wq(x)  # [B, S, D]
        k_proj = self.wk(x)
        v_proj = self.wv(x)

        # Reshape for multi-head
        q = q_proj.view(B, S, H, self.head_dim).transpose(1, 2)  # [B, H, S, D/H]
        k = k_proj.view(B, S, H, self.head_dim).transpose(1, 2)
        v = v_proj.view(B, S, H, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]
        scores_masked = scores + self.mask[:, :, :S, :S]
        attn_weights = F.softmax(scores_masked, dim=-1)

        # Attention output
        attn_out = torch.matmul(attn_weights, v)  # [B, H, S, D/H]
        attn_merged = attn_out.transpose(1, 2).contiguous().view(B, S, D)

        # Output projection
        output = self.wo(attn_merged)

        return output, {
            'q_proj': q_proj,
            'k_proj': k_proj,
            'v_proj': v_proj,
            'q': q,
            'k': k,
            'v': v,
            'scores': scores,
            'scores_masked': scores_masked,
            'attn_weights': attn_weights,
            'attn_out': attn_out,
            'attn_merged': attn_merged,
        }


def run_ttnn_attention(x_tt, wq, wk, wv, wo, mask_tt, heads: int, head_dim: int, scale: float, device):
    """Run attention using TTNN ops, return output and intermediates."""
    B = x_tt.shape[0]
    S = x_tt.shape[1]
    D = x_tt.shape[2]
    H = heads

    # QKV projections: y = x @ W.T (no bias)
    q_proj = ttnn.matmul(x_tt, wq, transpose_b=True)
    k_proj = ttnn.matmul(x_tt, wk, transpose_b=True)
    v_proj = ttnn.matmul(x_tt, wv, transpose_b=True)

    # Reshape for multi-head: [B, S, D] -> [B, H, S, D/H]
    q = ttnn.reshape(q_proj, (B, S, H, head_dim))
    q = ttnn.transpose(q, 1, 2)  # [B, H, S, D/H]
    k = ttnn.reshape(k_proj, (B, S, H, head_dim))
    k = ttnn.transpose(k, 1, 2)
    v = ttnn.reshape(v_proj, (B, S, H, head_dim))
    v = ttnn.transpose(v, 1, 2)

    # Attention scores: Q @ K.T * scale
    scores = ttnn.matmul(q, k, transpose_b=True)
    scores = ttnn.multiply(scores, scale)

    # Apply causal mask
    scores_masked = ttnn.add(scores, mask_tt)

    # Softmax
    attn_weights = ttnn.softmax(scores_masked, dim=-1)

    # Attention output
    attn_out = ttnn.matmul(attn_weights, v)  # [B, H, S, D/H]

    # Merge heads: [B, H, S, D/H] -> [B, S, D]
    attn_merged = ttnn.transpose(attn_out, 1, 2)  # [B, S, H, D/H]
    attn_merged = ttnn.reshape(attn_merged, (B, S, D))

    # Output projection
    output = ttnn.matmul(attn_merged, wo, transpose_b=True)

    return output, {
        'q_proj': q_proj,
        'k_proj': k_proj,
        'v_proj': v_proj,
        'q': q,
        'k': k,
        'v': v,
        'scores': scores,
        'scores_masked': scores_masked,
        'attn_weights': attn_weights,
        'attn_out': attn_out,
        'attn_merged': attn_merged,
    }


def init_weights_constant(model: nn.Module, value: float):
    """Initialize all weights to constant for reproducibility."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.fill_(value)


def main():
    print("=" * 70)
    print("Forward Pass Verification: TTNN vs PyTorch (BF16)")
    print("=" * 70)
    print(f"Config: batch={BATCH}, seq={SEQ}, dim={DIM}, heads={HEADS}")
    print(f"Seed: {SEED}")

    # Create PyTorch model (BF16)
    model = PyTorchAttention(DIM, HEADS, SEQ).bfloat16()
    init_weights_constant(model, 0.02)  # Same init as GPT-2 std

    # Create input (BF16) - use small values for stable attention
    x = torch.randn(BATCH, SEQ, DIM, dtype=torch.bfloat16) * 0.1

    # Run PyTorch forward
    print("\n--- PyTorch Forward ---")
    with torch.no_grad():
        output_pt, intermediates_pt = model(x)
    print(f"  Output shape: {list(output_pt.shape)}")
    print(f"  Output mean: {output_pt.float().mean().item():.6f}")
    print(f"  Output std: {output_pt.float().std().item():.6f}")

    # Open TTNN device
    print("\n--- TTNN Forward ---")
    device = ttnn.open_device(device_id=0)

    try:
        # Convert weights to TTNN
        wq_tt = ttnn.from_torch(model.wq.weight.clone(), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=device)
        wk_tt = ttnn.from_torch(model.wk.weight.clone(), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=device)
        wv_tt = ttnn.from_torch(model.wv.weight.clone(), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=device)
        wo_tt = ttnn.from_torch(model.wo.weight.clone(), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=device)

        # Convert input
        x_tt = ttnn.from_torch(x.clone(), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=device)

        # Create causal mask
        mask_tt = create_causal_mask_ttnn(SEQ, device)

        # Run TTNN forward
        scale = 1.0 / math.sqrt(HEAD_DIM)
        output_tt, intermediates_tt = run_ttnn_attention(
            x_tt, wq_tt, wk_tt, wv_tt, wo_tt, mask_tt,
            HEADS, HEAD_DIM, scale, device
        )

        # Convert output back to torch
        output_ttnn = ttnn.to_torch(output_tt)
        print(f"  Output shape: {list(output_ttnn.shape)}")
        print(f"  Output mean: {output_ttnn.float().mean().item():.6f}")
        print(f"  Output std: {output_ttnn.float().std().item():.6f}")

        # Compare intermediates
        print("\n--- Intermediate Comparisons ---")
        results = []

        # Q, K, V projections
        results.append(compare_tensors("q_proj",
            ttnn.to_torch(intermediates_tt['q_proj']), intermediates_pt['q_proj']))
        results.append(compare_tensors("k_proj",
            ttnn.to_torch(intermediates_tt['k_proj']), intermediates_pt['k_proj']))
        results.append(compare_tensors("v_proj",
            ttnn.to_torch(intermediates_tt['v_proj']), intermediates_pt['v_proj']))

        # Reshaped Q, K, V
        results.append(compare_tensors("q (reshaped)",
            ttnn.to_torch(intermediates_tt['q']), intermediates_pt['q']))
        results.append(compare_tensors("k (reshaped)",
            ttnn.to_torch(intermediates_tt['k']), intermediates_pt['k']))
        results.append(compare_tensors("v (reshaped)",
            ttnn.to_torch(intermediates_tt['v']), intermediates_pt['v']))

        # Attention scores
        results.append(compare_tensors("scores (Q@K.T * scale)",
            ttnn.to_torch(intermediates_tt['scores']), intermediates_pt['scores']))
        # Skip scores_masked comparison - PyTorch uses -inf, TTNN uses -1e9
        # They're functionally equivalent but numerically different
        print(f"  {'scores_masked':30s}: SKIPPED (PyTorch=-inf vs TTNN=-1e9)")
        results.append(compare_tensors("attn_weights (softmax)",
            ttnn.to_torch(intermediates_tt['attn_weights']), intermediates_pt['attn_weights']))

        # Attention output
        results.append(compare_tensors("attn_out (weights @ V)",
            ttnn.to_torch(intermediates_tt['attn_out']), intermediates_pt['attn_out']))
        results.append(compare_tensors("attn_merged",
            ttnn.to_torch(intermediates_tt['attn_merged']), intermediates_pt['attn_merged']))

        # Final output
        results.append(compare_tensors("OUTPUT (final)",
            output_ttnn, output_pt))

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        max_diffs = [r['max_diff'] for r in results]
        mean_diffs = [r['mean_diff'] for r in results]
        rel_errors = [r['rel_error'] for r in results]

        print(f"  Best case (min max_diff):  {min(max_diffs):.6f}")
        print(f"  Worst case (max max_diff): {max(max_diffs):.6f}")
        print(f"  Average mean_diff:         {sum(mean_diffs)/len(mean_diffs):.6f}")
        print(f"  Average rel_error:         {sum(rel_errors)/len(rel_errors):.4f}")

        # Find worst offender
        worst = max(results, key=lambda r: r['max_diff'])
        print(f"\n  Worst offender: {worst['name']} (max_diff={worst['max_diff']:.6f})")

        # Pass/fail thresholds (BF16 has ~3 decimal places precision)
        # max_diff < 0.1 is very strict, 0.05 is reasonable for BF16
        # rel_error < 0.15 (15%) is reasonable for accumulated matmul errors
        THRESH_MAX = 0.05
        THRESH_REL = 0.15

        all_pass = all(r['max_diff'] < THRESH_MAX and r['rel_error'] < THRESH_REL for r in results)

        if all_pass:
            print(f"\n  ✓ ALL PASS (max_diff < {THRESH_MAX}, rel_error < {THRESH_REL})")
        else:
            print(f"\n  ✗ SOME FAIL (threshold: max_diff < {THRESH_MAX}, rel_error < {THRESH_REL})")
            for r in results:
                if r['max_diff'] >= THRESH_MAX or r['rel_error'] >= THRESH_REL:
                    print(f"    FAIL: {r['name']}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
