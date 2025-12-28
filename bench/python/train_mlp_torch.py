#!/usr/bin/env python3
"""
MLP Training Experiment (PyTorch)
Compare with TTNN C++ implementation
"""

import torch
import torch.nn as nn
import time

def main():
    # Config - must match C++ exactly
    batch_size = 1024
    in_features = 512
    hidden = 256
    out_features = 128
    lr = 0.01
    num_iters = 100
    warmup_iters = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("MLP Training Experiment (PyTorch)")
    print("==================================")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Architecture: {in_features} -> {hidden} -> {out_features}")
    print(f"Learning rate: {lr}")
    print(f"Iterations: {num_iters} (warmup: {warmup_iters})")
    print()

    # Create model
    model = nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_features)
    ).to(device)

    # Initialize weights similar to C++ (uniform -0.1 to 0.1)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.1, 0.1)
            nn.init.zeros_(m.bias)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Fixed random input and target
    torch.manual_seed(42)
    x = torch.randn(batch_size, in_features, device=device)
    target = torch.randn(batch_size, out_features, device=device)

    # Training loop
    iter_times = []
    losses = []

    for i in range(num_iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        # Forward
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)

        # Backward
        loss.backward()

        # Optimizer step
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        ms = (end - start) * 1000

        if i >= warmup_iters:
            iter_times.append(ms)
        losses.append(loss.item())

        if i < 5 or i >= num_iters - 5 or i % 20 == 0:
            print(f"Iter {i:3d} | Loss: {loss.item():.6f} | Time: {ms:.2f} ms")

    # Statistics
    avg_time = sum(iter_times) / len(iter_times)

    print()
    print("==================================")
    print("Results:")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss:   {losses[-1]:.6f}")
    print(f"  Loss reduced: {'YES' if losses[0] > losses[-1] else 'NO'}")
    print(f"  Avg time/iter (after warmup): {avg_time:.2f} ms")
    print(f"  Throughput: {1000.0 / avg_time:.1f} iters/sec")


if __name__ == "__main__":
    main()
