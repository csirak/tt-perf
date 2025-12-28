#!/usr/bin/env python3
"""
Benchmark: 2-layer MLP training step (forward + backward + SGD) on PyTorch CPU
Matches the TTNN benchmark configuration for comparison.
"""

import time
import torch
import torch.nn as nn

# Benchmark configuration (matches TTNN)
BATCH = 1024
DIM = 512
WARMUP = 20
ITERS = 200
LR = 0.01


class MLP(nn.Module):
    """Simple 2-layer MLP with ReLU activation"""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, out_dim, bias=True)

        # Initialize weights to 0.01 (matches C++ init)
        with torch.no_grad():
            self.linear1.weight.fill_(0.01)
            self.linear1.bias.fill_(0.0)
            self.linear2.weight.fill_(0.01)
            self.linear2.bias.fill_(0.0)

    def forward(self, x):
        h = torch.relu(self.linear1(x))
        return self.linear2(h)


def bench_torch_cpu():
    """Benchmark PyTorch CPU"""
    device = torch.device("cpu")
    dtype = torch.float32  # CPU typically uses fp32

    # Create model
    model = MLP(DIM, DIM, DIM).to(device)

    # Create input and target
    x = torch.ones(BATCH, DIM, dtype=dtype, device=device)
    target = torch.full((BATCH, DIM), 0.5, dtype=dtype, device=device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # Warmup
    for _ in range(WARMUP):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

    # Timed iterations
    start = time.perf_counter()
    for _ in range(ITERS):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
    end = time.perf_counter()

    return (end - start) * 1000 / ITERS  # ms per iter


def bench_torch_cpu_bf16():
    """Benchmark PyTorch CPU with bfloat16 (if supported)"""
    device = torch.device("cpu")
    dtype = torch.bfloat16

    # Create model in fp32, then convert
    model = MLP(DIM, DIM, DIM).to(device).to(dtype)

    # Create input and target
    x = torch.ones(BATCH, DIM, dtype=dtype, device=device)
    target = torch.full((BATCH, DIM), 0.5, dtype=dtype, device=device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    # Warmup
    for _ in range(WARMUP):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

    # Timed iterations
    start = time.perf_counter()
    for _ in range(ITERS):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
    end = time.perf_counter()

    return (end - start) * 1000 / ITERS  # ms per iter


def main():
    print(f"PyTorch CPU Benchmark: 2-layer MLP")
    print("=" * 50)
    print(f"Config: batch={BATCH}, dim={DIM}, warmup={WARMUP}, iters={ITERS}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print()

    print("Running fp32 benchmark...")
    t_fp32 = bench_torch_cpu()
    print(f"  torch CPU (fp32): {t_fp32:.3f} ms/iter")

    print("Running bf16 benchmark...")
    try:
        t_bf16 = bench_torch_cpu_bf16()
        print(f"  torch CPU (bf16): {t_bf16:.3f} ms/iter")
    except Exception as e:
        print(f"  torch CPU (bf16): not supported ({e})")
        t_bf16 = None

    print()
    print("For comparison:")
    print("  TTNN Python dynamic:  9.506 ms/iter")
    print("  TTNN C++ dynamic:     0.984 ms/iter")
    print("  TTNN C++ static+trace: 0.563 ms/iter")

    print()
    if t_fp32 > 0:
        print(f"Speedup vs torch CPU (fp32):")
        print(f"  TTNN C++ dynamic:      {t_fp32 / 0.984:.2f}x faster")
        print(f"  TTNN C++ static+trace: {t_fp32 / 0.563:.2f}x faster")


if __name__ == "__main__":
    main()
