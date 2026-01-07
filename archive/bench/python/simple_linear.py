#!/usr/bin/env python3
"""Simple Linear layer for Tracy profiling."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import ttnn
from src.engine import Tensor
from src.nn import Linear


def main():
    torch.manual_seed(0)

    device = ttnn.open_device(device_id=0)
    try:
        # Simple 32x32 Linear layer
        x = torch.randn(32, 32)
        linear = Linear(32, 32, device=device)

        y = linear(x)
        y_t = y.to_torch().float()

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {tuple(y_t.shape)}")
        print(f"Weight shape: {linear.weight.to_torch().shape}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
