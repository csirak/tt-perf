#!/usr/bin/env python3
"""
Compare Tracy CSV outputs from static+trace vs traced+trace_api benchmarks.
Identifies differences in op counts and timing.

Usage:
    python3 scripts/compare_tracy.py static_trace_ops.csv traced_trace_ops.csv
"""

import sys
import csv
from collections import defaultdict


def parse_tracy_csv(filename):
    """Parse Tracy CSV export and return op counts/times by name."""
    ops = defaultdict(lambda: {"count": 0, "total_ns": 0})

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)

        # Find column indices
        name_idx = None
        time_idx = None
        for i, col in enumerate(header or []):
            if 'name' in col.lower():
                name_idx = i
            if 'time' in col.lower() or 'ns' in col.lower() or 'duration' in col.lower():
                time_idx = i

        if name_idx is None:
            print(f"Warning: Could not find 'name' column in {filename}")
            print(f"Header: {header}")
            return ops

        for row in reader:
            if len(row) <= name_idx:
                continue
            name = row[name_idx]
            ops[name]["count"] += 1
            if time_idx is not None and len(row) > time_idx:
                try:
                    ops[name]["total_ns"] += int(row[time_idx])
                except ValueError:
                    pass

    return ops


def compare_ops(static_ops, traced_ops):
    """Compare op dictionaries and print differences."""
    all_ops = set(static_ops.keys()) | set(traced_ops.keys())

    print("\n" + "=" * 70)
    print("Op Comparison: static+trace vs traced+trace_api")
    print("=" * 70)

    # Find ops unique to each
    static_only = set(static_ops.keys()) - set(traced_ops.keys())
    traced_only = set(traced_ops.keys()) - set(static_ops.keys())

    if static_only:
        print(f"\nOps ONLY in static+trace ({len(static_only)}):")
        for op in sorted(static_only):
            print(f"  {op}: count={static_ops[op]['count']}")

    if traced_only:
        print(f"\nOps ONLY in traced+trace_api ({len(traced_only)}):")
        for op in sorted(traced_only):
            print(f"  {op}: count={traced_ops[op]['count']}")

    # Compare common ops
    common = set(static_ops.keys()) & set(traced_ops.keys())
    print(f"\nCommon ops ({len(common)}):")
    print(f"{'Op Name':<40} {'static':<10} {'traced':<10} {'diff':<10}")
    print("-" * 70)

    total_static = 0
    total_traced = 0

    for op in sorted(common):
        s_count = static_ops[op]["count"]
        t_count = traced_ops[op]["count"]
        diff = s_count - t_count

        s_time = static_ops[op]["total_ns"]
        t_time = traced_ops[op]["total_ns"]

        total_static += s_time
        total_traced += t_time

        diff_str = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else "="
        print(f"{op:<40} {s_count:<10} {t_count:<10} {diff_str:<10}")

    print("-" * 70)

    # Summary
    static_total_count = sum(op["count"] for op in static_ops.values())
    traced_total_count = sum(op["count"] for op in traced_ops.values())

    print(f"\nSummary:")
    print(f"  static+trace total ops: {static_total_count}")
    print(f"  traced+trace total ops: {traced_total_count}")
    print(f"  Difference: {static_total_count - traced_total_count}")

    if total_static > 0 and total_traced > 0:
        print(f"\n  static+trace total time: {total_static / 1e6:.3f} ms")
        print(f"  traced+trace total time: {total_traced / 1e6:.3f} ms")
        print(f"  Time difference: {(total_static - total_traced) / 1e6:.3f} ms")


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 compare_tracy.py <static_trace.csv> <traced_trace.csv>")
        sys.exit(1)

    static_file = sys.argv[1]
    traced_file = sys.argv[2]

    print(f"Loading {static_file}...")
    static_ops = parse_tracy_csv(static_file)

    print(f"Loading {traced_file}...")
    traced_ops = parse_tracy_csv(traced_file)

    compare_ops(static_ops, traced_ops)


if __name__ == "__main__":
    main()
