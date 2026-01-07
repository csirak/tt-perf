#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Profile individual GPT-2 ops to understand time breakdown

import ttnn
import torch
import time

# GPT-2 mini config
BATCH, SEQ, DIM, HEADS = 32, 256, 512, 8
FFN_DIM = DIM * 4
HEAD_DIM = DIM // HEADS
M = BATCH * SEQ  # 8192

device = ttnn.open_device(device_id=0)

def make_tensor(shape):
    t = torch.randn(shape, dtype=torch.bfloat16)
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Create test tensors
x = make_tensor((BATCH, SEQ, DIM))
wq = make_tensor((DIM, DIM))
w1 = make_tensor((DIM, FFN_DIM))
w2 = make_tensor((FFN_DIM, DIM))
ln_gamma = make_tensor((1, DIM))
ln_beta = make_tensor((1, DIM))
h_expanded = make_tensor((BATCH, SEQ, FFN_DIM))
scores = make_tensor((BATCH, HEADS, SEQ, SEQ))
q_4d = make_tensor((BATCH, HEADS, SEQ, HEAD_DIM))
k_4d = make_tensor((BATCH, HEADS, SEQ, HEAD_DIM))
v_4d = make_tensor((BATCH, HEADS, SEQ, HEAD_DIM))

print("Warmup...")
for _ in range(5):
    _ = ttnn.matmul(x, wq, transpose_b=True)
    _ = ttnn.matmul(x, w1, transpose_b=True)
ttnn.synchronize_device(device)

def time_op(name, fn, n=20):
    ttnn.synchronize_device(device)
    start = time.time()
    for _ in range(n):
        fn()
    ttnn.synchronize_device(device)
    return (time.time() - start) / n * 1000

results = []

# Matmul ops with FLOPs
results.append(("QKV proj [8192,512]@[512,512]", time_op("qkv", lambda: ttnn.matmul(x, wq, transpose_b=True)), 2*M*DIM*DIM))
results.append(("FFN up [8192,512]@[512,2048]", time_op("ffn1", lambda: ttnn.matmul(x, w1, transpose_b=True)), 2*M*DIM*FFN_DIM))
results.append(("FFN down [8192,2048]@[2048,512]", time_op("ffn2", lambda: ttnn.matmul(h_expanded, w2, transpose_b=True)), 2*M*FFN_DIM*DIM))

# Attention matmuls
attn_scores_flops = 2 * BATCH * HEADS * SEQ * HEAD_DIM * SEQ
results.append(("Attn scores [B,H,S,D/H]@[B,H,D/H,S]", time_op("attn_s", lambda: ttnn.matmul(q_4d, k_4d, transpose_b=True)), attn_scores_flops))
results.append(("Attn out [B,H,S,S]@[B,H,S,D/H]", time_op("attn_o", lambda: ttnn.matmul(scores, v_4d)), attn_scores_flops))

# SDPA (fused)
results.append(("SDPA (fused)", time_op("sdpa", lambda: ttnn.transformer.scaled_dot_product_attention(q_4d, k_4d, v_4d, is_causal=True)), 2*attn_scores_flops))

# Non-matmul ops (no FLOPs, memory-bound)
results.append(("LayerNorm [B,S,D]", time_op("ln", lambda: ttnn.layer_norm(x, epsilon=1e-5, weight=ln_gamma, bias=ln_beta)), 0))
results.append(("GELU [B,S,FFN]", time_op("gelu", lambda: ttnn.gelu(h_expanded)), 0))
results.append(("Softmax [B,H,S,S]", time_op("sm", lambda: ttnn.softmax(scores, dim=-1)), 0))
results.append(("Add [B,S,D]", time_op("add", lambda: ttnn.add(x, x)), 0))

print()
print("=" * 90)
print("GPT-2 Per-Op Timing and TFLOPS")
print(f"Config: batch={BATCH}, seq={SEQ}, dim={DIM}, heads={HEADS}, ffn_mult=4")
print("=" * 90)
print(f"{'Operation':<40} {'Time(ms)':>10} {'GFLOPs':>10} {'TFLOPS':>10}")
print("-" * 90)

total_time = 0
total_flops = 0
for name, t, flops in results:
    tflops = flops / (t/1000) / 1e12 if flops > 0 and t > 0 else 0
    gflops = flops / 1e9
    print(f"{name:<40} {t:>10.3f} {gflops:>10.2f} {tflops:>10.2f}")
    total_time += t
    total_flops += flops

print("-" * 90)
print(f"{'TOTAL (matmul ops only)':<40} {total_time:>10.3f} {total_flops/1e9:>10.2f}")

# Per-layer estimate
print()
print("=" * 90)
print("Per-Layer Estimate (forward only)")
print("=" * 90)
qkv_time = results[0][1] * 3  # Q, K, V
wo_time = results[0][1]       # Output proj same shape
ffn_up = results[1][1]
ffn_down = results[2][1]
ln_time = results[6][1] * 2   # 2 layer norms
gelu_time = results[7][1]
add_time = results[9][1] * 2  # 2 residuals

# Using SDPA instead of separate attention matmuls
sdpa_time = results[5][1]

forward_time = qkv_time + sdpa_time + wo_time + ln_time + ffn_up + gelu_time + ffn_down + add_time
print(f"QKV projections (3x): {qkv_time:.3f} ms")
print(f"SDPA:                 {sdpa_time:.3f} ms")
print(f"Output projection:    {wo_time:.3f} ms")
print(f"LayerNorm (2x):       {ln_time:.3f} ms")
print(f"FFN up:               {ffn_up:.3f} ms")
print(f"GELU:                 {gelu_time:.3f} ms")
print(f"FFN down:             {ffn_down:.3f} ms")
print(f"Add residuals (2x):   {add_time:.3f} ms")
print(f"{'='*40}")
print(f"Forward per layer:    {forward_time:.3f} ms")
print(f"Backward (~2x):       {forward_time*2:.3f} ms")
print(f"Total per layer:      {forward_time*3:.3f} ms")
print(f"6 layers:             {forward_time*3*6:.3f} ms")
print()
print(f"Actual benchmark: 406 ms")
print(f"Estimated from ops: {forward_time*3*6:.1f} ms")

ttnn.close_device(device)
