#!/usr/bin/env python3
"""
Distributed matmul benchmark on 2-device N300 (1x2 mesh).

Pattern: Column-parallel tensor parallelism
- Replicate A to both devices
- Shard B by columns (dim=1) across devices
- Each device computes half the output columns
- all_gather to combine results

This actually distributes work across both devices, unlike replicating
both tensors which runs redundant computation on each device.
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


def benchmark_distributed_matmul(mesh_device, size: int, include_allgather: bool = True) -> dict:
    """
    Benchmark distributed matmul with column-parallel sharding.

    Args:
        mesh_device: 1x2 MeshDevice
        size: Matrix dimension (must be divisible by 32 for tiles and 2 for sharding)
        include_allgather: Whether to include all_gather in timing (True for e2e)

    Returns:
        dict with timing stats and throughput
    """
    # Create input tensors
    a = torch.randn(size, size, dtype=torch.bfloat16)
    b = torch.randn(size, size, dtype=torch.bfloat16)

    # Replicate A to both devices
    a_tt = ttnn.from_torch(
        a,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Shard B by columns (dim=1) - each device gets half the columns
    b_tt = ttnn.from_torch(
        b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )

    # Warmup
    for _ in range(N_WARMUP):
        c_sharded = ttnn.matmul(a_tt, b_tt)
        if include_allgather:
            c_full = ttnn.all_gather(
                c_sharded,
                dim=1,
                num_links=1,
                topology=ttnn.Topology.Linear,
            )
        ttnn.synchronize_device(mesh_device)

    # Timed runs
    times = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        c_sharded = ttnn.matmul(a_tt, b_tt)
        if include_allgather:
            c_full = ttnn.all_gather(
                c_sharded,
                dim=1,
                num_links=1,
                topology=ttnn.Topology.Linear,
            )
        ttnn.synchronize_device(mesh_device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # microseconds

    times = np.array(times)

    # FLOPS: 2 * N^3 for the full matmul (split across 2 devices)
    # Since we're doing the equivalent of a full N×N × N×N matmul
    flops = 2.0 * (size ** 3)
    tflops = flops / (times.mean() * 1e-6) / 1e12

    return {
        "size": size,
        "mean_us": times.mean(),
        "std_us": times.std(),
        "min_us": times.min(),
        "max_us": times.max(),
        "tflops": tflops,
    }


def benchmark_single_device_matmul(mesh_device, size: int) -> dict:
    """
    Benchmark single-device matmul for comparison.
    Uses only device 0 in the mesh.
    """
    a = torch.randn(size, size, dtype=torch.bfloat16)
    b = torch.randn(size, size, dtype=torch.bfloat16)

    # Get first device from mesh
    device = mesh_device.get_device(0)

    a_tt = ttnn.from_torch(
        a,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )
    b_tt = ttnn.from_torch(
        b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )

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
        times.append((t1 - t0) * 1e6)

    times = np.array(times)
    flops = 2.0 * (size ** 3)
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
    """Generate test sizes (must be divisible by 32 for tiles and 64 for 2-way sharding)."""
    sizes = []

    # Powers of 2 (all are divisible by 64)
    for exp in range(6, 13):  # 64 to 4096
        sizes.append(2 ** exp)

    # Additional sizes around inflection points (divisible by 64)
    additional = [
        192, 256, 320, 384, 448, 512, 576, 640,  # Early region
        896, 960, 1024, 1088, 1152,               # Core saturation
        1920, 1984, 2048, 2112, 2176,             # Memory boundary
        3840, 3904, 3968, 4032, 4096,             # Large size
    ]

    for s in additional:
        if s not in sizes and s % 64 == 0:
            sizes.append(s)

    return sorted(set(sizes))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Distributed matmul benchmark on N300")
    parser.add_argument("--mode", choices=["distributed", "single", "both"], default="both",
                        help="Benchmark mode: distributed, single-device, or both")
    parser.add_argument("--no-allgather", action="store_true",
                        help="Skip all_gather in distributed mode (just measure matmul)")
    parser.add_argument("--sizes", type=str, default=None,
                        help="Comma-separated list of sizes to test")
    args = parser.parse_args()

    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",")]
    else:
        sizes = generate_test_sizes()

    print(f"# Distributed matmul benchmark on 1x2 MeshDevice (N300)", file=sys.stderr)
    print(f"# Mode: {args.mode}, include_allgather: {not args.no_allgather}", file=sys.stderr)
    print(f"# Testing {len(sizes)} sizes", file=sys.stderr)

    # Open 1x2 mesh device
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

    try:
        if args.mode in ["distributed", "both"]:
            print("\n# Distributed (column-parallel) matmul")
            print("mode,size,mean_us,std_us,min_us,max_us,tflops")

            for size in sizes:
                try:
                    result = benchmark_distributed_matmul(
                        mesh_device, size,
                        include_allgather=not args.no_allgather
                    )
                    print(f"distributed,{result['size']},{result['mean_us']:.2f},"
                          f"{result['std_us']:.2f},{result['min_us']:.2f},"
                          f"{result['max_us']:.2f},{result['tflops']:.3f}")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"# Error at distributed size {size}: {e}", file=sys.stderr)
                    import traceback
                    traceback.print_exc()

        if args.mode in ["single", "both"]:
            print("\n# Single-device matmul (for comparison)")
            print("mode,size,mean_us,std_us,min_us,max_us,tflops")

            for size in sizes:
                try:
                    result = benchmark_single_device_matmul(mesh_device, size)
                    print(f"single,{result['size']},{result['mean_us']:.2f},"
                          f"{result['std_us']:.2f},{result['min_us']:.2f},"
                          f"{result['max_us']:.2f},{result['tflops']:.3f}")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"# Error at single size {size}: {e}", file=sys.stderr)

    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
