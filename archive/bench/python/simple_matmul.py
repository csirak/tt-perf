#!/usr/bin/env python3
"""Simple matmul for Tracy profiling."""
import torch
import ttnn

def main():
    torch.manual_seed(0)

    # Simple 32x32 matmul
    a = torch.randn(1, 1, 32, 32)
    b = torch.randn(1, 1, 32, 32)

    with ttnn.manage_device(0) as device:
        a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
        b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
        c_tt = ttnn.matmul(a_tt, b_tt)
        c = ttnn.to_torch(c_tt)

    print(f"Result shape: {tuple(c.shape)}")
    print(f"Max error: {(c - torch.matmul(a, b)).abs().max().item():.6f}")

if __name__ == "__main__":
    main()
