#!/usr/bin/env python3
"""
Matmul sweep benchmark across multiple sizes using TTNN Python API.
Compares against NumPy (OpenBLAS) baseline.
"""

import os
import sys
import time
import numpy as np

# Setup TT-Metal environment
if "TT_METAL_HOME" not in os.environ:
    os.environ["TT_METAL_HOME"] = os.path.expanduser("~/tt-metal")
if "ARCH_NAME" not in os.environ:
    os.environ["ARCH_NAME"] = "wormhole_b0"

import torch
import ttnn

N_WARMUP = 10
N_ITER = 50


def benchmark_ttnn(device, size: int) -> dict:
    """Benchmark TTNN matmul for given size."""
    a = torch.ones(size, size, dtype=torch.bfloat16)
    b = torch.ones(size, size, dtype=torch.bfloat16)

    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    # Warmup
    for _ in range(N_WARMUP):
        c = ttnn.matmul(a_tt, b_tt)
        ttnn.synchronize_device(device)

    # Timed runs
    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        c = ttnn.matmul(a_tt, b_tt)
        ttnn.synchronize_device(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # Convert to microseconds

    times = np.array(times)
    return {
        "size": size,
        "mean_us": times.mean(),
        "std_us": times.std(),
        "min_us": times.min(),
        "max_us": times.max(),
    }


def benchmark_numpy(size: int) -> dict:
    """Benchmark NumPy (OpenBLAS) matmul for given size."""
    a = np.random.randn(size, size).astype(np.float32)
    b = np.random.randn(size, size).astype(np.float32)

    # Warmup
    for _ in range(N_WARMUP):
        _ = a @ b

    # Timed runs
    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        _ = a @ b
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    times = np.array(times)
    return {
        "size": size,
        "mean_us": times.mean(),
        "std_us": times.std(),
        "min_us": times.min(),
        "max_us": times.max(),
    }


def benchmark_torch(size: int) -> dict:
    """Benchmark PyTorch (CPU) matmul for given size."""
    a = torch.randn(size, size, dtype=torch.float32)
    b = torch.randn(size, size, dtype=torch.float32)

    # Warmup
    for _ in range(N_WARMUP):
        _ = torch.matmul(a, b)

    # Timed runs
    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        _ = torch.matmul(a, b)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)

    times = np.array(times)
    return {
        "size": size,
        "mean_us": times.mean(),
        "std_us": times.std(),
        "min_us": times.min(),
        "max_us": times.max(),
    }


def main():
    sizes = [128, 256, 512, 1024, 2048, 4096]

    print("backend,size,mean_us,std_us,min_us,max_us")

    # TTNN Python benchmark
    with ttnn.manage_device(device_id=0) as device:
        for size in sizes:
            result = benchmark_ttnn(device, size)
            print(f"ttnn_python,{result['size']},{result['mean_us']:.2f},"
                  f"{result['std_us']:.2f},{result['min_us']:.2f},{result['max_us']:.2f}")

    # NumPy benchmark
    for size in sizes:
        result = benchmark_numpy(size)
        print(f"numpy,{result['size']},{result['mean_us']:.2f},"
              f"{result['std_us']:.2f},{result['min_us']:.2f},{result['max_us']:.2f}")

    # PyTorch benchmark
    for size in sizes:
        result = benchmark_torch(size)
        print(f"torch,{result['size']},{result['mean_us']:.2f},"
              f"{result['std_us']:.2f},{result['min_us']:.2f},{result['max_us']:.2f}")


if __name__ == "__main__":
    main()
