#!/usr/bin/env python3
"""
Matmul inflection point analysis using 2x1 MeshDevice on n300.
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

N_WARMUP = 5
N_ITER = 20


def benchmark_ttnn_mesh(mesh_device, size: int) -> dict:
    """Benchmark TTNN matmul on 1x2 mesh device."""
    a = torch.ones(size, size, dtype=torch.bfloat16)
    b = torch.ones(size, size, dtype=torch.bfloat16)

    # Replicate tensors to all devices in mesh
    a_tt = ttnn.from_torch(
        a,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    b_tt = ttnn.from_torch(
        b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Warmup
    for _ in range(N_WARMUP):
        c = ttnn.matmul(a_tt, b_tt)
        ttnn.synchronize_device(mesh_device)

    # Timed runs
    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        c = ttnn.matmul(a_tt, b_tt)
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds

    times = np.array(times)

    # Calculate TFLOP/s - 2x work on 2 devices
    flops = 2 * 2 * (size ** 3)
    tflops = flops / (times.mean() * 1e-6) / 1e12

    return {
        "size": size,
        "mean_us": times.mean(),
        "std_us": times.std(),
        "min_us": times.min(),
        "max_us": times.max(),
        "tflops": tflops,
    }


def generate_test_sizes():
    """Generate strategic test sizes focusing on inflection points."""
    sizes = set()

    # Coarse sweep: powers of 2
    for exp in range(5, 13):  # 32 to 4096
        sizes.add(2 ** exp)

    # Fine sweep around suspected inflection points
    inflection_regions = [
        (192, 384, 32),    # Early region
        (384, 640, 32),    # L1/tile boundary
        (896, 1152, 32),   # Core saturation
        (1920, 2176, 32),  # Memory boundary
        (3840, 4096, 64),  # Large size boundary
    ]

    for start, end, step in inflection_regions:
        for s in range(start, end + 1, step):
            if s % 32 == 0:
                sizes.add(s)

    return sorted(sizes)


def main():
    sizes = generate_test_sizes()

    print(f"# Testing {len(sizes)} sizes on 1x2 MeshDevice (n300)", file=sys.stderr)
    print("size,mean_us,std_us,min_us,max_us,tflops")

    # Try without fabric config - just basic mesh device
    # The fabric config seems to cause issues with control plane mapping
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

    try:
        for size in sizes:
            try:
                result = benchmark_ttnn_mesh(mesh_device, size)
                print(f"{result['size']},{result['mean_us']:.2f},"
                      f"{result['std_us']:.2f},{result['min_us']:.2f},"
                      f"{result['max_us']:.2f},{result['tflops']:.3f}")
                sys.stdout.flush()
            except Exception as e:
                print(f"# Error at size {size}: {e}", file=sys.stderr)
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
