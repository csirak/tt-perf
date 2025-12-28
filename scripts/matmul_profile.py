#!/usr/bin/env python3
import os
import sys
import time
import argparse
import csv
from typing import List, Tuple

DEFAULT_TT_METAL_HOME = "/home/howard/tt-metal"
DEFAULT_OUT_PATH = "/home/howard/mod-arith/profile.csv"

if "TT_METAL_HOME" not in os.environ:
    os.environ["TT_METAL_HOME"] = DEFAULT_TT_METAL_HOME
if "ARCH_NAME" not in os.environ:
    os.environ["ARCH_NAME"] = "wormhole_b0"
if DEFAULT_TT_METAL_HOME not in sys.path:
    sys.path.insert(0, DEFAULT_TT_METAL_HOME)

import torch
import ttnn


def parse_triples(raw: str) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Bad triple: {item}. Expected M,N,K")
        m, n, k = (int(parts[0]), int(parts[1]), int(parts[2]))
        triples.append((m, n, k))
    return triples


def parse_sizes(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def ensure_tile_multiple(name: str, val: int) -> None:
    if val % 32 != 0:
        raise ValueError(f"{name}={val} must be a multiple of 32 for TILE_LAYOUT")


def time_matmul(device, a_tt, b_tt, repeats: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = ttnn.matmul(a_tt, b_tt)
        ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    return t1 - t0


def time_softmax(device, x_tt, repeats: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = ttnn.softmax(x_tt, dim=-1)
        ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    return t1 - t0


def time_total(device, a_tt, b_tt, repeats: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        c_tt = ttnn.matmul(a_tt, b_tt)
        s_tt = ttnn.softmax(c_tt, dim=-1)
        _ = ttnn.to_torch(s_tt)
        ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    return t1 - t0


def main() -> int:
    ap = argparse.ArgumentParser(description="TTNN matmul + softmax ablation with timing")
    ap.add_argument("--triples", default="", help="M,N,K triples: '256,256,256;512,512,512'")
    ap.add_argument("--sizes", default="128,256,512", help="Square sizes for M=N=K")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=DEFAULT_OUT_PATH, help="CSV output path")
    ap.add_argument("--append", action="store_true", help="Append to output file")
    ap.add_argument("--verbose", action="store_true", help="Print status to stdout")
    args = ap.parse_args()

    if args.triples:
        triples = parse_triples(args.triples)
    else:
        sizes = parse_sizes(args.sizes)
        triples = [(s, s, s) for s in sizes]

    dtype_map = {
        "bf16": (torch.bfloat16, ttnn.bfloat16),
        "fp32": (torch.float32, ttnn.float32),
    }
    torch_dtype, ttnn_dtype = dtype_map[args.dtype]

    rows = []
    with ttnn.manage_device(0) as device:
        for (m, n, k) in triples:
            ensure_tile_multiple("M", m)
            ensure_tile_multiple("N", n)
            ensure_tile_multiple("K", k)

            a = torch.randn(args.batch, args.heads, m, k, dtype=torch_dtype)
            b = torch.randn(args.batch, args.heads, k, n, dtype=torch_dtype)

            a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)
            b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn_dtype)

            for _ in range(args.warmup):
                _ = ttnn.matmul(a_tt, b_tt)
                ttnn.synchronize_device(device)

            c_tt = ttnn.matmul(a_tt, b_tt)
            for _ in range(args.warmup):
                _ = ttnn.softmax(c_tt, dim=-1)
                ttnn.synchronize_device(device)

            for _ in range(args.warmup):
                c_tt = ttnn.matmul(a_tt, b_tt)
                s_tt = ttnn.softmax(c_tt, dim=-1)
                _ = ttnn.to_torch(s_tt)
                ttnn.synchronize_device(device)

            matmul_total = time_matmul(device, a_tt, b_tt, args.repeats)
            softmax_total = time_softmax(device, c_tt, args.repeats)
            total_total = time_total(device, a_tt, b_tt, args.repeats)

            data_total = max(0.0, total_total - (matmul_total + softmax_total))

            matmul_avg_ms = (matmul_total / args.repeats) * 1000.0
            softmax_avg_ms = (softmax_total / args.repeats) * 1000.0
            total_avg_ms = (total_total / args.repeats) * 1000.0
            data_avg_ms = (data_total / args.repeats) * 1000.0

            rows.append((
                m, n, k, args.batch, args.heads,
                matmul_avg_ms,
                softmax_avg_ms,
                total_avg_ms,
                data_avg_ms,
            ))

    header = (
        "M", "N", "K", "batch", "heads",
        "matmul_avg_ms", "softmax_avg_ms", "total_avg_ms", "data_avg_ms",
    )

    out_path = args.out
    mode = "a" if args.append else "w"
    write_header = (mode == "w")

    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(list(header))
        for r in rows:
            writer.writerow([
                r[0], r[1], r[2], r[3], r[4],
                f"{r[5]:.3f}", f"{r[6]:.3f}", f"{r[7]:.3f}", f"{r[8]:.3f}",
            ])

    if args.verbose:
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
