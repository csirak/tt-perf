#!/usr/bin/env python3
# Compare activation norm TSVs (C++ vs Torch) and emit diffs + plots.

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_tsv(path: Path) -> Tuple[List[int], Dict[str, Dict[str, List[float]]]]:
    lines = path.read_text().strip().splitlines()
    if not lines:
        return [], {}
    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:]]

    step_idx = header.index("step")
    steps = [int(r[step_idx]) for r in rows]

    metrics: Dict[str, Dict[str, List[float]]] = {}
    for col, name in enumerate(header):
        if name == "step":
            continue
        if name.endswith("_l2"):
            metric = name[:-3]
            metrics.setdefault(metric, {}).setdefault("l2", [])
            for r in rows:
                metrics[metric]["l2"].append(float(r[col]))
        elif name.endswith("_l1"):
            metric = name[:-3]
            metrics.setdefault(metric, {}).setdefault("l1", [])
            for r in rows:
                metrics[metric]["l1"].append(float(r[col]))

    # Keep only metrics with both l2/l1
    metrics = {k: v for k, v in metrics.items() if "l2" in v and "l1" in v}
    return steps, metrics


def rel_diff(a: float, b: float, eps: float = 1e-6) -> float:
    denom = abs(a)
    if denom == 0.0:
        return 0.0 if abs(b) == 0.0 else float("inf")
    return abs(b - a) / (denom + eps)


def write_compare(out_path: Path, steps: List[int], cpp: Dict[str, Dict[str, List[float]]],
                  pt: Dict[str, Dict[str, List[float]]]) -> None:
    lines = ["step\tmetric\tcpp_l2\ttorch_l2\tabs_diff_l2\trel_diff_l2\tcpp_l1\ttorch_l1\tabs_diff_l1\trel_diff_l1"]
    for metric in sorted(cpp.keys()):
        if metric not in pt:
            continue
        c_l2 = cpp[metric]["l2"]
        c_l1 = cpp[metric]["l1"]
        p_l2 = pt[metric]["l2"]
        p_l1 = pt[metric]["l1"]
        for i, step in enumerate(steps):
            l2_cpp = c_l2[i]
            l2_pt = p_l2[i]
            l1_cpp = c_l1[i]
            l1_pt = p_l1[i]
            lines.append(
                f"{step}\t{metric}\t{l2_cpp:.6g}\t{l2_pt:.6g}\t{abs(l2_pt - l2_cpp):.6g}\t{rel_diff(l2_cpp, l2_pt):.6g}"
                f"\t{l1_cpp:.6g}\t{l1_pt:.6g}\t{abs(l1_pt - l1_cpp):.6g}\t{rel_diff(l1_cpp, l1_pt):.6g}"
            )
    out_path.write_text("\n".join(lines) + "\n")


def write_summary(out_path: Path, steps: List[int], cpp: Dict[str, Dict[str, List[float]]],
                  pt: Dict[str, Dict[str, List[float]]]) -> Dict[str, float]:
    lines = ["metric\tmax_rel_l2\tstep_max_rel_l2\tmax_rel_l1\tstep_max_rel_l1"]
    max_rel_l2 = {}
    for metric in sorted(cpp.keys()):
        if metric not in pt:
            continue
        c_l2 = cpp[metric]["l2"]
        c_l1 = cpp[metric]["l1"]
        p_l2 = pt[metric]["l2"]
        p_l1 = pt[metric]["l1"]
        rels_l2 = [rel_diff(c_l2[i], p_l2[i]) for i in range(len(steps))]
        rels_l1 = [rel_diff(c_l1[i], p_l1[i]) for i in range(len(steps))]
        max_l2 = max(rels_l2) if rels_l2 else 0.0
        max_l1 = max(rels_l1) if rels_l1 else 0.0
        step_l2 = steps[rels_l2.index(max_l2)] if rels_l2 else -1
        step_l1 = steps[rels_l1.index(max_l1)] if rels_l1 else -1
        lines.append(f"{metric}\t{max_l2:.6g}\t{step_l2}\t{max_l1:.6g}\t{step_l1}")
        max_rel_l2[metric] = max_l2
    out_path.write_text("\n".join(lines) + "\n")
    return max_rel_l2


def plot_top_rel(out_path: Path, steps: List[int], cpp: Dict[str, Dict[str, List[float]]],
                 pt: Dict[str, Dict[str, List[float]]], top_metrics: List[str], label: str) -> None:
    plt.figure(figsize=(10, 6))
    for metric in top_metrics:
        if metric not in cpp or metric not in pt:
            continue
        rels = [rel_diff(cpp[metric]["l2"][i], pt[metric]["l2"][i]) for i in range(len(steps))]
        plt.plot(steps, rels, marker="o", linewidth=1.5, label=metric)
    plt.xlabel("step")
    plt.ylabel("rel_diff_l2")
    plt.title(f"Top {len(top_metrics)} rel_diff_l2: {label}")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare C++ vs Torch activation norms")
    parser.add_argument("--cpp", type=str, default="experiments/grok/outputs/steps/act_norms_cpp.tsv")
    parser.add_argument("--torch", type=str, default="experiments/grok/outputs/steps/act_norms_torch.tsv")
    parser.add_argument("--out-compare", type=str, default="experiments/grok/outputs/steps/act_norms_compare.tsv")
    parser.add_argument("--out-summary", type=str, default="experiments/grok/outputs/steps/act_norms_summary.tsv")
    parser.add_argument("--out-plot", type=str, default="experiments/grok/outputs/steps/act_norms_top_rel_l2.png")
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()

    cpp_path = Path(args.cpp)
    pt_path = Path(args.torch)
    steps_cpp, cpp = load_tsv(cpp_path)
    steps_pt, pt = load_tsv(pt_path)

    if steps_cpp != steps_pt:
        common_steps = sorted(set(steps_cpp) & set(steps_pt))
        def filter_steps(steps, metrics):
            idx_map = [steps.index(s) for s in common_steps]
            for m in metrics.values():
                for k in ["l2", "l1"]:
                    m[k] = [m[k][i] for i in idx_map]
            return common_steps
        steps = filter_steps(steps_cpp, cpp)
        steps = filter_steps(steps_pt, pt)
    else:
        steps = steps_cpp

    Path(args.out_compare).parent.mkdir(parents=True, exist_ok=True)
    write_compare(Path(args.out_compare), steps, cpp, pt)
    max_rel_l2 = write_summary(Path(args.out_summary), steps, cpp, pt)

    top_metrics = sorted(max_rel_l2.keys(), key=lambda k: max_rel_l2[k], reverse=True)[:args.top]
    plot_top_rel(Path(args.out_plot), steps, cpp, pt, top_metrics, "CPP vs Torch")
    print(f"Wrote {args.out_compare}")
    print(f"Wrote {args.out_summary}")
    print(f"Wrote {args.out_plot}")


if __name__ == "__main__":
    main()
