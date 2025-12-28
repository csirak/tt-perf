#!/usr/bin/env python3
"""
Analyze NoC traces for core usage patterns and throughput metrics.

Key metrics:
- Per-op core usage (which cores are active)
- Core heatmap (event distribution across cores)
- Throughput per core (bytes/core, to understand scaling efficiency)
- DRAM vs core-to-core traffic breakdown
"""

import json
import os
import sys
from glob import glob
from collections import defaultdict

def analyze_trace_dir(trace_dir):
    """Analyze all NoC trace files in a directory."""
    results = {
        'ops': [],                    # list of {op_id, num_cores, cores, total_bytes}
        'core_heatmap': defaultdict(int),  # (x,y) -> event_count
        'core_bytes': defaultdict(int),    # (x,y) -> total_bytes sent
        'traffic_matrix': defaultdict(int), # ((sx,sy),(dx,dy)) -> total_bytes
        'dram_reads': 0,              # bytes read from DRAM
        'dram_writes': 0,             # bytes written to DRAM
        'core_to_core': 0,            # bytes between compute cores
        'total_bytes': 0,
    }

    # DRAM banks are at y=0 and y=11 on Wormhole
    DRAM_Y = {0, 11}

    trace_files = sorted(glob(f"{trace_dir}/noc_trace_dev0_ID*.json"))
    if not trace_files:
        print(f"No trace files found in {trace_dir}")
        return results

    for trace_file in trace_files:
        # Extract op ID from filename
        fname = os.path.basename(trace_file)
        op_id = int(fname.split('ID')[1].replace('.json', ''))

        with open(trace_file) as f:
            events = json.load(f)

        src_cores = set()
        op_bytes = 0

        for e in events:
            if 'sx' not in e:
                continue

            sx, sy = e['sx'], e['sy']
            src_cores.add((sx, sy))
            results['core_heatmap'][(sx, sy)] += 1

            if 'num_bytes' in e and 'dx' in e:
                num_bytes = e['num_bytes']
                dx, dy = e['dx'], e['dy']

                op_bytes += num_bytes
                results['total_bytes'] += num_bytes
                results['core_bytes'][(sx, sy)] += num_bytes
                results['traffic_matrix'][((sx, sy), (dx, dy))] += num_bytes

                # Categorize traffic
                if dy in DRAM_Y:
                    if 'READ' in e.get('type', ''):
                        results['dram_reads'] += num_bytes
                    else:
                        results['dram_writes'] += num_bytes
                elif sy not in DRAM_Y and dy not in DRAM_Y:
                    results['core_to_core'] += num_bytes

        results['ops'].append({
            'op_id': op_id,
            'num_cores': len(src_cores),
            'cores': sorted(src_cores),
            'total_bytes': op_bytes,
            'bytes_per_core': op_bytes / len(src_cores) if src_cores else 0,
        })

    return results

def print_report(results, title="NoC Analysis"):
    """Print a formatted analysis report."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

    # Categorize ops by core count
    ops_by_cores = defaultdict(list)
    for op in results['ops']:
        ops_by_cores[op['num_cores']].append(op)

    print(f"\n## Op Summary ({len(results['ops'])} total ops)")
    print("-" * 40)
    for num_cores in sorted(ops_by_cores.keys()):
        ops = ops_by_cores[num_cores]
        total_bytes = sum(op['total_bytes'] for op in ops)
        avg_bytes_per_core = total_bytes / (num_cores * len(ops)) if ops else 0
        print(f"  {num_cores:2d} cores: {len(ops):3d} ops, "
              f"{total_bytes/1e6:8.2f} MB total, "
              f"{avg_bytes_per_core/1e3:8.2f} KB/core avg")

    # Throughput per core analysis
    print(f"\n## Throughput Per Core")
    print("-" * 40)
    total_cores_used = len(results['core_bytes'])
    total_bytes = results['total_bytes']
    avg_bytes_per_core = total_bytes / total_cores_used if total_cores_used else 0

    print(f"  Total bytes transferred: {total_bytes/1e6:.2f} MB")
    print(f"  Unique cores active: {total_cores_used}")
    print(f"  Avg bytes per core: {avg_bytes_per_core/1e3:.2f} KB")

    # Per-core breakdown (top 10)
    sorted_cores = sorted(results['core_bytes'].items(), key=lambda x: -x[1])
    print(f"\n  Top 10 cores by traffic:")
    for (x, y), bytes_sent in sorted_cores[:10]:
        print(f"    Core ({x},{y}): {bytes_sent/1e3:8.2f} KB")

    # Traffic breakdown
    print(f"\n## Traffic Breakdown")
    print("-" * 40)
    print(f"  DRAM reads:    {results['dram_reads']/1e6:8.2f} MB ({100*results['dram_reads']/max(1,total_bytes):.1f}%)")
    print(f"  DRAM writes:   {results['dram_writes']/1e6:8.2f} MB ({100*results['dram_writes']/max(1,total_bytes):.1f}%)")
    print(f"  Core-to-core:  {results['core_to_core']/1e6:8.2f} MB ({100*results['core_to_core']/max(1,total_bytes):.1f}%)")

    # Core grid visualization
    print(f"\n## Core Activity Heatmap (event count)")
    print("-" * 40)
    print_core_heatmap(results['core_heatmap'])

    # Scaling efficiency summary
    print(f"\n## Scaling Efficiency Summary")
    print("-" * 40)
    for num_cores in sorted(ops_by_cores.keys()):
        ops = ops_by_cores[num_cores]
        total_bytes = sum(op['total_bytes'] for op in ops)
        bytes_per_core = total_bytes / (num_cores * len(ops)) if ops else 0

        # Compare to theoretical: if we had perfect scaling,
        # bytes_per_core should be constant regardless of num_cores
        print(f"  {num_cores:2d} cores: {bytes_per_core/1e3:8.2f} KB/core/op")

def print_core_heatmap(heatmap, max_x=10, max_y=12):
    """Print ASCII heatmap of core activity."""
    if not heatmap:
        print("  (no data)")
        return

    max_count = max(heatmap.values())

    # Header
    print("     ", end="")
    for x in range(max_x):
        print(f" {x:2d}", end="")
    print()
    print("    +" + "---" * max_x)

    for y in range(max_y):
        print(f" {y:2d} |", end="")
        for x in range(max_x):
            count = heatmap.get((x, y), 0)
            if count == 0:
                char = " . "
            else:
                # Normalize to 0-9 scale
                level = int(9 * count / max_count)
                char = f" {level} "
            print(char, end="")
        print()

def compare_configs(results_a, results_b, name_a="Config A", name_b="Config B"):
    """Compare two configurations for scaling efficiency."""
    print(f"\n{'='*60}")
    print(f" Comparison: {name_a} vs {name_b}")
    print('='*60)

    # Get bytes per core for each config
    def get_bytes_per_core(results):
        total_bytes = results['total_bytes']
        total_cores = len(results['core_bytes'])
        return total_bytes / total_cores if total_cores else 0

    bpc_a = get_bytes_per_core(results_a)
    bpc_b = get_bytes_per_core(results_b)

    print(f"\n## Throughput Per Core Comparison")
    print("-" * 40)
    print(f"  {name_a}: {bpc_a/1e3:.2f} KB/core")
    print(f"  {name_b}: {bpc_b/1e3:.2f} KB/core")
    if bpc_a > 0:
        print(f"  Ratio: {bpc_b/bpc_a:.2f}x")

    # Core utilization comparison
    cores_a = len(results_a['core_bytes'])
    cores_b = len(results_b['core_bytes'])
    print(f"\n## Core Utilization")
    print("-" * 40)
    print(f"  {name_a}: {cores_a} cores active")
    print(f"  {name_b}: {cores_b} cores active")

def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_noc_cores.py <trace_dir> [trace_dir2]")
        print("  Single dir: analyze and report")
        print("  Two dirs: analyze both and compare")
        sys.exit(1)

    trace_dir = os.path.expanduser(sys.argv[1])
    results = analyze_trace_dir(trace_dir)
    print_report(results, f"Analysis: {os.path.basename(trace_dir)}")

    if len(sys.argv) >= 3:
        trace_dir2 = os.path.expanduser(sys.argv[2])
        results2 = analyze_trace_dir(trace_dir2)
        print_report(results2, f"Analysis: {os.path.basename(trace_dir2)}")
        compare_configs(results, results2,
                       os.path.basename(trace_dir),
                       os.path.basename(trace_dir2))

if __name__ == "__main__":
    main()
