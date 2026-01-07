#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tensor I/O utilities for cross-implementation verification.
# Binary format: [ndim:u32][dim0:u32][dim1:u32]...[data:bf16[]]

import struct
import torch
import numpy as np
from pathlib import Path


def save_tensor(t: torch.Tensor, path: str) -> None:
    """Save tensor to binary file in BF16 format.

    Format:
        - 4 bytes: ndim (uint32)
        - 4 bytes * ndim: dimensions (uint32 each)
        - N * 2 bytes: data (bfloat16)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to BF16 if not already
    if t.dtype != torch.bfloat16:
        t = t.bfloat16()

    # Make contiguous
    t = t.contiguous()

    with open(path, 'wb') as f:
        # Write ndim
        f.write(struct.pack('<I', t.ndim))

        # Write dimensions
        for dim in t.shape:
            f.write(struct.pack('<I', dim))

        # Write data as raw bytes (BF16 = 2 bytes per element)
        # PyTorch BF16 is stored as uint16 internally
        # Use numpy to reinterpret the bytes
        data = t.view(torch.int16).numpy().view(np.uint16).tobytes()
        f.write(data)


def load_tensor(path: str) -> torch.Tensor:
    """Load tensor from binary file.

    Returns:
        torch.Tensor in bfloat16 dtype
    """
    with open(path, 'rb') as f:
        # Read ndim
        ndim = struct.unpack('<I', f.read(4))[0]

        # Read dimensions
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack('<I', f.read(4))[0])

        # Calculate total elements
        numel = 1
        for d in shape:
            numel *= d

        # Read data
        data = f.read(numel * 2)  # 2 bytes per bf16

        # Convert to tensor
        # Read as uint16, reinterpret as int16 for torch compatibility
        arr = np.frombuffer(data, dtype=np.uint16).astype(np.int16)
        t = torch.from_numpy(arr.copy()).view(torch.bfloat16)

        return t.reshape(shape)


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor) -> dict:
    """Compare two tensors and return error metrics."""
    a_f = a.float().flatten()
    b_f = b.float().flatten()

    diff = (a_f - b_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # Relative error (avoid div by zero)
    denom = b_f.abs().clamp(min=1e-6)
    rel_err = (diff / denom).mean().item()

    # Percentiles
    p50 = diff.median().item()
    p99 = diff.quantile(0.99).item() if len(diff) > 100 else max_diff

    return {
        'name': name,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'p50': p50,
        'p99': p99,
        'rel_error': rel_err,
        'shape': list(a.shape),
    }


def print_comparison(result: dict) -> None:
    """Print comparison result."""
    name = result['name']
    print(f"  {name:30s}: max={result['max_diff']:.6f}, mean={result['mean_diff']:.6f}, "
          f"p99={result['p99']:.6f}, rel={result['rel_error']:.4f}")


if __name__ == "__main__":
    # Test round-trip
    print("Testing tensor I/O...")

    # Create test tensor
    t = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    print(f"Original: shape={list(t.shape)}, dtype={t.dtype}")
    print(f"Values: {t.flatten()[:5].tolist()}")

    # Save
    save_tensor(t, "/tmp/test_tensor.bin")

    # Load
    t2 = load_tensor("/tmp/test_tensor.bin")
    print(f"Loaded: shape={list(t2.shape)}, dtype={t2.dtype}")
    print(f"Values: {t2.flatten()[:5].tolist()}")

    # Compare
    result = compare_tensors("round_trip", t, t2)
    print_comparison(result)

    assert result['max_diff'] == 0.0, "Round-trip should be exact!"
    print("PASS: Round-trip exact match")
