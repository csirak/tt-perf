#!/usr/bin/env python3
import os
import sys
import time
import argparse
import csv
from typing import List

DEFAULT_TT_METAL_HOME = "/home/howard/tt-metal"
DEFAULT_TTNN_PY_PATH = "/home/howard/tt-metal/ttnn"
DEFAULT_OUT_PATH = "/home/howard/mod-arith/q_profile.csv"

if "TT_METAL_HOME" not in os.environ:
    os.environ["TT_METAL_HOME"] = DEFAULT_TT_METAL_HOME
if "ARCH_NAME" not in os.environ:
    os.environ["ARCH_NAME"] = "wormhole_b0"
if DEFAULT_TTNN_PY_PATH not in sys.path:
    sys.path.insert(0, DEFAULT_TTNN_PY_PATH)
if DEFAULT_TT_METAL_HOME not in sys.path:
    sys.path.insert(0, DEFAULT_TT_METAL_HOME)

import torch
import ttnn


def parse_sizes(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def run_mode(
    device,
    count: int,
    m: int,
    n: int,
    k: int,
    batch: int,
    heads: int,
    torch_dtype,
    ttnn_dtype,
    mode: str,
    data_cq: int,
    compute_cq: int,
    a_list,
    b_list,
) -> float:
    t0 = time.perf_counter()
    if mode == "sync_each":
        for i in range(count):
            a_tt = ttnn.from_torch(a_list[i], layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype, cq_id=data_cq)
            b_tt = ttnn.from_torch(b_list[i], layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype, cq_id=data_cq)
            _ = ttnn.matmul(a_tt, b_tt, queue_id=compute_cq)
            ttnn.synchronize_device(device)
    elif mode == "queued":
        for i in range(count):
            a_tt = ttnn.from_torch(a_list[i], layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype, cq_id=data_cq)
            b_tt = ttnn.from_torch(b_list[i], layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype, cq_id=data_cq)
            _ = ttnn.matmul(a_tt, b_tt, queue_id=compute_cq)
        ttnn.synchronize_device(device)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    t1 = time.perf_counter()
    return t1 - t0


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare queued vs sync-each matmul wall time (includes host->device copy)")
    ap.add_argument("--sizes", default="1024", help="Comma-separated square sizes for M=N=K")
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--count", type=int, default=8, help="Number of matmuls to enqueue (min 8)")
    ap.add_argument("--warmup", type=int, default=2, help="Warmup iterations (min 2)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--num-queues", type=int, default=2)
    ap.add_argument("--data-cq", type=int, default=0)
    ap.add_argument("--compute-cq", type=int, default=1)
    ap.add_argument("--out", default=DEFAULT_OUT_PATH, help="CSV output path")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    sizes = parse_sizes(args.sizes) if args.m is None and args.n is None and args.k is None else []

    dtype_map = {
        "bf16": (torch.bfloat16, ttnn.bfloat16),
        "fp32": (torch.float32, ttnn.float32),
    }
    torch_dtype, ttnn_dtype = dtype_map[args.dtype]

    if args.count < 8:
        raise ValueError("--count must be >= 8")
    if args.warmup < 2:
        raise ValueError("--warmup must be >= 2")

    header = [
        "mode", "M", "N", "K", "count", "wall_s", "avg_ms_per_matmul", "speedup_pct"
    ]

    out_path = args.out
    mode = "w"
    write_header = True

    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        def run_for_size(m, n, k):
            if m % 32 or n % 32 or k % 32:
                raise ValueError("M, N, K must be multiples of 32 for TILE_LAYOUT")

            if args.num_queues < 1:
                raise ValueError("--num-queues must be >= 1")
            if args.data_cq < 0 or args.data_cq >= args.num_queues:
                raise ValueError("--data-cq must be within [0, num-queues)")
            if args.compute_cq < 0 or args.compute_cq >= args.num_queues:
                raise ValueError("--compute-cq must be within [0, num-queues)")

            device = ttnn.CreateDevice(device_id=0, num_command_queues=args.num_queues)
            try:
                warmup_a = [torch.randn(args.batch, args.heads, m, k, dtype=torch_dtype) for _ in range(args.warmup)]
                warmup_b = [torch.randn(args.batch, args.heads, k, n, dtype=torch_dtype) for _ in range(args.warmup)]
                for _ in range(args.warmup):
                    a_tt = ttnn.from_torch(warmup_a[_], layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype, cq_id=args.data_cq)
                    b_tt = ttnn.from_torch(warmup_b[_], layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype, cq_id=args.data_cq)
                    _ = ttnn.matmul(a_tt, b_tt, queue_id=args.compute_cq)
                    ttnn.synchronize_device(device)

                a_list = [torch.randn(args.batch, args.heads, m, k, dtype=torch_dtype) for _ in range(args.count)]
                b_list = [torch.randn(args.batch, args.heads, k, n, dtype=torch_dtype) for _ in range(args.count)]

                sync_each_s = run_mode(
                    device,
                    args.count,
                    m,
                    n,
                    k,
                    args.batch,
                    args.heads,
                    torch_dtype,
                    ttnn_dtype,
                    "sync_each",
                    args.data_cq,
                    args.compute_cq,
                    a_list,
                    b_list,
                )
                queued_s = run_mode(
                    device,
                    args.count,
                    m,
                    n,
                    k,
                    args.batch,
                    args.heads,
                    torch_dtype,
                    ttnn_dtype,
                    "queued",
                    args.data_cq,
                    args.compute_cq,
                    a_list,
                    b_list,
                )
            finally:
                ttnn.close_device(device)

            if sync_each_s > 0:
                speedup_pct = ((sync_each_s - queued_s) / sync_each_s) * 100.0
            else:
                speedup_pct = 0.0

            rows = [
                ["sync_each", m, n, k, args.count,
                 f"{sync_each_s:.6f}", f"{(sync_each_s / args.count) * 1000.0:.3f}", "0.000"],
                ["queued", m, n, k, args.count,
                 f"{queued_s:.6f}", f"{(queued_s / args.count) * 1000.0:.3f}", f"{speedup_pct:.3f}"],
            ]
            writer.writerows(rows)

        if sizes:
            for s in sizes:
                run_for_size(s, s, s)
        else:
            m = args.m
            n = args.n
            k = args.k
            if m is None or n is None or k is None:
                raise ValueError("Either --sizes or all of --m/--n/--k must be provided")
            run_for_size(m, n, k)

    if args.verbose:
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
