#!/usr/bin/env python3
"""
MLP Benchmark Visualization
Generates a heatmap and table from benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def main():
    # Find CSV file
    csv_path = 'bench_mlp_results.csv'
    if not os.path.exists(csv_path):
        csv_path = '../bench_mlp_results.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Cannot find {csv_path}")
        sys.exit(1)

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} benchmark results from {csv_path}\n")

    # Pivot to 2D table
    pivot = df.pivot(index='layers', columns='dim', values='ms_per_iter')

    # Print table
    print("=" * 60)
    print("MLP Training Step Time (ms/iter) - Traced Execution")
    print("=" * 60)
    print(f"{'layers':<8}", end='')
    for dim in pivot.columns:
        print(f"{dim:>10}", end='')
    print()
    print("-" * 60)
    for layers in pivot.index:
        print(f"{layers:<8}", end='')
        for dim in pivot.columns:
            val = pivot.loc[layers, dim]
            print(f"{val:>10.3f}", end='')
        print()
    print("=" * 60)
    print()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use log scale for better visualization since values span large range
    data = pivot.values
    im = ax.imshow(data, cmap='viridis', aspect='auto')

    # Labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Hidden Dimension', fontsize=12)
    ax.set_ylabel('Number of Layers', fontsize=12)
    ax.set_title('MLP Training Step Time (ms/iter) - Traced Execution\nBatch=1024, All layers same dimension', fontsize=14)

    # Annotate cells with values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            # Use white text for dark cells, black for light
            color = 'white' if val < data.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}',
                    ha='center', va='center', color=color, fontsize=11, fontweight='bold')

    cbar = plt.colorbar(im, label='ms/iter')
    plt.tight_layout()

    # Save
    output_path = 'bench_mlp_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to {output_path}")

    # Also create a scaling analysis
    print("\n" + "=" * 60)
    print("Scaling Analysis")
    print("=" * 60)

    # Compute scaling factors
    print("\nDimension scaling (relative to 256):")
    for layers in pivot.index:
        base = pivot.loc[layers, 256]
        print(f"  {layers} layers: ", end='')
        for dim in pivot.columns:
            ratio = pivot.loc[layers, dim] / base
            print(f"{dim}={ratio:.2f}x  ", end='')
        print()

    print("\nLayer scaling (relative to 2 layers):")
    for dim in pivot.columns:
        base = pivot.loc[2, dim]
        print(f"  dim={dim}: ", end='')
        for layers in pivot.index:
            ratio = pivot.loc[layers, dim] / base
            print(f"{layers}L={ratio:.2f}x  ", end='')
        print()

    # Show plot
    plt.show()

if __name__ == '__main__':
    main()
