#!/usr/bin/env python3
# Compare C++ vs PyTorch dumps and write TSV with forward + grad errors.

from pathlib import Path
import sys
import torch

sys.path.append("/home/howard/ttnn-perf/experiments/grok")
import tensor_io as tio

base = Path("/home/howard/ttnn-perf/experiments/grok/outputs")
cpp_dir = base / "cpp"
pt_dir = base / "pytorch"
out_path = base / "compare.tsv"

cpp_bins = {p.name for p in cpp_dir.glob("*.bin")}
pt_bins = {p.name for p in pt_dir.glob("*.bin")}
common = sorted(cpp_bins & pt_bins)

rows = []
forward_diffs = []
forward_refs = []
forward_abs_diffs = []
forward_abs_refs = []

def squeeze_leading_ones(t):
    shape = list(t.shape)
    idx = 0
    while idx < len(shape) and shape[idx] == 1:
        idx += 1
    if idx == 0:
        return t
    return t.reshape(shape[idx:]) if shape[idx:] else t.reshape([1])

def align_tensors(cpp, pt):
    # squeeze leading singleton dims
    cpp_s = squeeze_leading_ones(cpp)
    pt_s = squeeze_leading_ones(pt)
    if list(cpp_s.shape) == list(pt_s.shape):
        return cpp_s, pt_s

    # if one has extra leading dims that match tail of the other, sum-reduce leading dims
    if len(cpp_s.shape) > len(pt_s.shape):
        tail = list(cpp_s.shape[-len(pt_s.shape):])
        if tail == list(pt_s.shape):
            lead = len(cpp_s.shape) - len(pt_s.shape)
            cpp_s = cpp_s.float()
            for dim in range(lead):
                cpp_s = cpp_s.sum(dim=0)
            return cpp_s, pt_s
    if len(pt_s.shape) > len(cpp_s.shape):
        tail = list(pt_s.shape[-len(cpp_s.shape):])
        if tail == list(cpp_s.shape):
            lead = len(pt_s.shape) - len(cpp_s.shape)
            pt_s = pt_s.float()
            for dim in range(lead):
                pt_s = pt_s.sum(dim=0)
            return cpp_s, pt_s

    return cpp_s, pt_s

def is_forward_tensor(name):
    if name.endswith("_grad.bin"):
        return False
    if any(tag in name for tag in ["_weight", "_bias", "_gamma", "_beta"]):
        return False
    if name in ["tok_weight.bin", "pos_weight.bin", "output_weight.bin", "output_bias.bin"]:
        return False
    return True

def format_rel(num, denom):
    if denom == 0.0:
        return "0/0" if num == 0.0 else "inf"
    return f"{num / denom:.6g}"

def format_rel_eps(num, denom, eps):
    return f"{num / (denom + eps):.6g}"

for name in common:
    cpp = tio.load_tensor(cpp_dir / name)
    pt = tio.load_tensor(pt_dir / name)
    shape = "x".join(str(d) for d in cpp.shape)
    cpp_a, pt_a = align_tensors(cpp, pt)
    if tuple(cpp_a.shape) != tuple(pt_a.shape):
        rows.append((
            name,
            shape,
            "shape_mismatch",
            "shape_mismatch",
            "shape_mismatch",
            "shape_mismatch",
            "shape_mismatch",
            "shape_mismatch",
            "shape_mismatch",
            "shape_mismatch",
            "shape_mismatch",
            "shape_mismatch",
        ))
        continue
    diff = (pt_a.float() - cpp_a.float()).flatten()
    ref = cpp_a.float().flatten()

    l2 = float(torch.linalg.norm(diff))
    l2_ref = float(torch.linalg.norm(ref))
    l1 = float(torch.sum(torch.abs(diff)))
    l1_ref = float(torch.sum(torch.abs(ref)))
    max_ref = float(torch.max(ref))
    max_pt = float(torch.max(pt_a.float().flatten()))
    max_diff = float(torch.max(diff))

    rel_l2 = format_rel(l2, l2_ref)
    rel_l1 = format_rel(l1, l1_ref)
    rel_l2_eps = format_rel_eps(l2, l2_ref, 1e-6)

    dot = float(torch.dot(pt_a.float().flatten(), cpp_a.float().flatten()))
    denom = float(torch.linalg.norm(pt_a.float().flatten()) * torch.linalg.norm(cpp_a.float().flatten()))
    cos_sim = format_rel(dot, denom)
    if is_forward_tensor(name):
        forward_diffs.append(diff)
        forward_refs.append(ref)
        forward_abs_diffs.append(torch.abs(diff))
        forward_abs_refs.append(torch.abs(ref))

    rows.append((
        name,
        shape,
        f"{l2:.6g}",
        rel_l2,
        rel_l2_eps,
        f"{l1:.6g}",
        rel_l1,
        f"{max_ref:.6g}",
        f"{max_pt:.6g}",
        f"{max_diff:.6g}",
        cos_sim,
    ))

# total forward error across all forward tensors (concatenated)
if forward_diffs:
    all_diff = torch.cat(forward_diffs)
    all_ref = torch.cat(forward_refs)
    all_abs_diff = torch.cat(forward_abs_diffs)
    all_abs_ref = torch.cat(forward_abs_refs)
    total_l2 = float(torch.linalg.norm(all_diff))
    ref_l2 = float(torch.linalg.norm(all_ref))
    total_l1 = float(torch.sum(all_abs_diff))
    ref_l1 = float(torch.sum(all_abs_ref))
    total_max_ref = float(torch.max(all_ref))
    all_pt = all_ref + all_diff
    total_max_pt = float(torch.max(all_pt))
    total_max_diff = float(torch.max(all_diff))
    dot = float(torch.dot(all_ref, all_pt))
    denom = float(torch.linalg.norm(all_ref) * torch.linalg.norm(all_pt))
    cos_sim = format_rel(dot, denom)
    rows.append((
        "__total_forward__",
        "all",
        f"{total_l2:.6g}",
        format_rel(total_l2, ref_l2),
        format_rel_eps(total_l2, ref_l2, 1e-6),
        f"{total_l1:.6g}",
        format_rel(total_l1, ref_l1),
        f"{total_max_ref:.6g}",
        f"{total_max_pt:.6g}",
        f"{total_max_diff:.6g}",
        cos_sim,
    ))

# sort by rel err desc where numeric, keep total at end

def rel_key(r):
    if r[0] == "__total_forward__":
        return -1
    try:
        return float(r[3])
    except Exception:
        return -1

rows_sorted = sorted([r for r in rows if r[0] != "__total_forward__"], key=rel_key, reverse=True)
if any(r[0] == "__total_forward__" for r in rows):
    rows_sorted.append([r for r in rows if r[0] == "__total_forward__"][0])

lines = []
lines.append("tensor\tshape\tabs_err_L2\trel_err_L2\trel_err_L2_eps\tabs_err_L1\trel_err_L1\tmax_ref\tmax_pt\tmax_diff\tcos_sim")
for r in rows_sorted:
    lines.append("\t".join(r))

out_path.write_text("\n".join(lines) + "\n")
print(f"Wrote {out_path}")
