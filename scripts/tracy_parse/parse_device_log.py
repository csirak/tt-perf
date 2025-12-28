#!/usr/bin/env python3
"""
Parse Tracy device profiler logs to extract parallel kernel execution times.

The device profiler outputs raw cycle data per core/RISC. This script calculates
the actual wall-clock time by finding the timeline bounds (earliest start to
latest end) since kernels execute in parallel across cores.

Usage:
    python parse_device_log.py [path_to_profile_log_device.csv]

Default path: ~/tt-metal/generated/profiler/.logs/profile_log_device.csv
"""

import csv
import sys
import os
from collections import defaultdict
from pathlib import Path


def parse_device_log(csv_path: str) -> dict:
    """
    Parse device profile CSV and extract parallel execution times.

    Returns dict with:
        - run_timings: per-run timing data
        - summary: overall summary stats
    """
    with open(csv_path) as f:
        reader = csv.reader(f)

        # First line is ARCH info
        arch_line = next(reader)
        arch_info = arch_line[0] if arch_line else "Unknown"

        # Second line is column headers
        next(reader)

        # Track absolute start/end times per run
        run_bounds = defaultdict(lambda: [float('inf'), float('-inf')])

        # Track per-RISC-type bounds for breakdown
        risc_bounds = defaultdict(lambda: defaultdict(lambda: [float('inf'), float('-inf')]))

        # Count cores per run
        cores_per_run = defaultdict(set)

        for row in reader:
            if len(row) < 12:
                continue

            core_x = row[1].strip()
            core_y = row[2].strip()
            risc = row[3].strip()
            cycles_str = row[5].strip()
            run_id = row[7].strip()
            zone_name = row[10].strip()

            try:
                cycles = int(cycles_str)
            except ValueError:
                continue

            # Only track kernel zones (not FW overhead)
            if 'KERNEL' not in zone_name:
                continue

            # Track cores
            cores_per_run[run_id].add((core_x, core_y))

            # Update global bounds for this run
            run_bounds[run_id][0] = min(run_bounds[run_id][0], cycles)
            run_bounds[run_id][1] = max(run_bounds[run_id][1], cycles)

            # Update per-RISC bounds
            if 'BRISC' in risc:
                risc_type = 'BRISC'
            elif 'NCRISC' in risc:
                risc_type = 'NCRISC'
            else:
                risc_type = 'TRISC'

            risc_bounds[run_id][risc_type][0] = min(risc_bounds[run_id][risc_type][0], cycles)
            risc_bounds[run_id][risc_type][1] = max(risc_bounds[run_id][risc_type][1], cycles)

    # Process results
    run_timings = {}
    for run_id in sorted(run_bounds.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        start, end = run_bounds[run_id]
        total_ns = end - start

        risc_data = {}
        for risc in ['BRISC', 'NCRISC', 'TRISC']:
            if risc in risc_bounds[run_id]:
                r_start, r_end = risc_bounds[run_id][risc]
                risc_data[risc] = {
                    'duration_ns': r_end - r_start,
                    'offset_ns': r_start - start,
                }

        run_timings[run_id] = {
            'total_ns': total_ns,
            'num_cores': len(cores_per_run[run_id]),
            'risc_breakdown': risc_data,
        }

    return {
        'arch_info': arch_info,
        'run_timings': run_timings,
    }


def print_report(data: dict):
    """Print formatted report of parallel execution times."""
    print(f"=== Device Profiler Analysis ===")
    print(f"{data['arch_info']}")
    print()

    run_timings = data['run_timings']
    run_ids = list(run_timings.keys())

    if not run_ids:
        print("No kernel data found.")
        return

    print(f"Found {len(run_ids)} runs\n")

    for run_id in run_ids:
        timing = run_timings[run_id]
        total_us = timing['total_ns'] / 1000

        print(f"Run {run_id}: {timing['num_cores']} cores")
        print(f"  Total parallel time: {total_us:.2f} µs")

        # Show RISC breakdown
        risc_data = timing['risc_breakdown']
        max_dur = max(r['duration_ns'] for r in risc_data.values()) if risc_data else 1

        for risc in ['BRISC', 'NCRISC', 'TRISC']:
            if risc in risc_data:
                rd = risc_data[risc]
                dur_us = rd['duration_ns'] / 1000
                offset_us = rd['offset_ns'] / 1000

                # ASCII bar
                bar_len = int(rd['duration_ns'] / max_dur * 40)
                offset_len = int(rd['offset_ns'] / max_dur * 40) if max_dur > 0 else 0
                bar = ' ' * offset_len + '=' * bar_len

                print(f"  {risc:6s}: |{bar:<40}| {dur_us:.2f} µs")
        print()

    # Summary for last run
    last_run = run_ids[-1]
    timing = run_timings[last_run]

    print("=== Last Run Summary ===")
    print(f"Total parallel execution: {timing['total_ns']/1000:.2f} µs")

    risc_data = timing['risc_breakdown']
    bottleneck = max(risc_data.keys(), key=lambda r: risc_data[r]['duration_ns'])
    print(f"Bottleneck: {bottleneck} ({risc_data[bottleneck]['duration_ns']/1000:.2f} µs)")

    # Check if compute or memory bound
    compute_time = risc_data.get('TRISC', {}).get('duration_ns', 0)
    dm_time = max(
        risc_data.get('BRISC', {}).get('duration_ns', 0),
        risc_data.get('NCRISC', {}).get('duration_ns', 0)
    )

    if dm_time > compute_time:
        print("Status: Memory-bound (data movement > compute)")
    else:
        print("Status: Compute-bound (compute > data movement)")


def main():
    # Default path
    default_path = os.path.expanduser(
        "~/tt-metal/generated/profiler/.logs/profile_log_device.csv"
    )

    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        print(f"Usage: {sys.argv[0]} [path_to_profile_log_device.csv]")
        sys.exit(1)

    data = parse_device_log(csv_path)
    print_report(data)


if __name__ == "__main__":
    main()
