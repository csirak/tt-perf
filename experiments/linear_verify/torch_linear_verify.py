#!/usr/bin/env python3
# BF16 PyTorch verification for linear, linear+layernorm, and linear+DyT stages.

import argparse
import json
from pathlib import Path

import torch
import sys

THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR.parent / "grok"))

from tensor_io import load_tensor, save_tensor, compare_tensors, print_comparison


def load_meta(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def layer_norm(x, gamma, beta, eps, dtype):
    x_f = x.float()
    mean = x_f.mean(dim=-1, keepdim=True)
    centered = x_f - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    x_norm = centered * rstd
    out = gamma.float() * x_norm + beta.float()
    return out.to(dtype)


def layer_norm_bf16(x, gamma, beta, eps, dtype):
    x_b = x.to(dtype)
    mean = x_b.mean(dim=-1, keepdim=True)
    centered = x_b - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    rstd = (var + eps).rsqrt()
    x_norm = centered * rstd
    out = gamma.to(dtype) * x_norm + beta.to(dtype)
    return out.to(dtype)


def dyt(x, alpha, gamma, beta, dtype):
    alpha_scalar = alpha.reshape(-1)[0].to(dtype)
    out = torch.tanh(x.to(dtype) * alpha_scalar)
    out = out * gamma.to(dtype) + beta.to(dtype)
    return out.to(dtype)


def save_cpu(t: torch.Tensor, path: Path) -> None:
    save_tensor(t.detach().cpu(), str(path))


def main():
    parser = argparse.ArgumentParser(description="Linear verify (BF16)")
    parser.add_argument("--cpp-dir", type=str, default="experiments/linear_verify/outputs/cpp")
    parser.add_argument("--out-dir", type=str, default="experiments/linear_verify/outputs/pytorch")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    cpp_dir = Path(args.cpp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(cpp_dir / "meta.json")
    batch = int(meta["batch_size"])
    seq = int(meta["seq"])
    in_dim = int(meta["in_dim"])
    out_dim = int(meta["out_dim"])
    stage = int(meta["stage"])
    eps = float(meta.get("eps", 1e-5))

    dtype = torch.bfloat16

    x = load_tensor(str(cpp_dir / "input.bin")).to(device=device, dtype=dtype)
    x = x.view(batch, seq, in_dim).detach().requires_grad_(True)
    target = load_tensor(str(cpp_dir / "target.bin")).to(device=device, dtype=dtype)
    target = target.view(batch, seq, out_dim)

    weight = load_tensor(str(cpp_dir / "weight.bin")).to(device=device, dtype=dtype).requires_grad_()
    bias = load_tensor(str(cpp_dir / "bias.bin")).to(device=device, dtype=dtype).requires_grad_()

    linear_out = (torch.matmul(x, weight.t()) + bias).to(dtype)
    linear_out.retain_grad()

    ln_out_from_cpp = None
    ln_out_from_cpp_bf16 = None
    dyt_out_from_cpp = None
    if stage == 1:
        ln_gamma = load_tensor(str(cpp_dir / "ln_gamma.bin")).to(device=device, dtype=dtype).requires_grad_()
        ln_beta = load_tensor(str(cpp_dir / "ln_beta.bin")).to(device=device, dtype=dtype).requires_grad_()
        ln_out = layer_norm(linear_out, ln_gamma, ln_beta, eps, dtype)
        ln_out.retain_grad()

        # Also compute LN on the C++ linear_out to isolate LN vs matmul differences
        cpp_linear_out = load_tensor(str(cpp_dir / "linear_out.bin")).to(device=device, dtype=dtype)
        cpp_linear_out = cpp_linear_out.view(batch, seq, out_dim)
        ln_out_from_cpp = layer_norm(cpp_linear_out, ln_gamma, ln_beta, eps, dtype)
        ln_out_from_cpp_bf16 = layer_norm_bf16(cpp_linear_out, ln_gamma, ln_beta, eps, dtype)

        out = ln_out
    elif stage == 2:
        dyt_alpha = load_tensor(str(cpp_dir / "dyt_alpha.bin")).to(device=device, dtype=dtype).requires_grad_()
        dyt_gamma = load_tensor(str(cpp_dir / "dyt_gamma.bin")).to(device=device, dtype=dtype).requires_grad_()
        dyt_beta = load_tensor(str(cpp_dir / "dyt_beta.bin")).to(device=device, dtype=dtype).requires_grad_()
        dyt_out = dyt(linear_out, dyt_alpha, dyt_gamma, dyt_beta, dtype)
        dyt_out.retain_grad()

        # Also compute DyT on the C++ linear_out to isolate DyT vs matmul differences
        cpp_linear_out = load_tensor(str(cpp_dir / "linear_out.bin")).to(device=device, dtype=dtype)
        cpp_linear_out = cpp_linear_out.view(batch, seq, out_dim)
        dyt_out_from_cpp = dyt(cpp_linear_out, dyt_alpha, dyt_gamma, dyt_beta, dtype)

        out = dyt_out
    else:
        out = linear_out

    diff = (out - target).to(dtype)
    loss = (diff * diff).mean().to(dtype)

    save_cpu(x, out_dir / "input.bin")
    save_cpu(target, out_dir / "target.bin")
    save_cpu(weight, out_dir / "weight.bin")
    save_cpu(bias, out_dir / "bias.bin")
    save_cpu(linear_out, out_dir / "linear_out.bin")
    if stage == 1:
        save_cpu(ln_gamma, out_dir / "ln_gamma.bin")
        save_cpu(ln_beta, out_dir / "ln_beta.bin")
        save_cpu(ln_out, out_dir / "ln_out.bin")
        if ln_out_from_cpp is not None:
            save_cpu(ln_out_from_cpp, out_dir / "ln_out_from_cpp_linear.bin")
        if ln_out_from_cpp_bf16 is not None:
            save_cpu(ln_out_from_cpp_bf16, out_dir / "ln_out_from_cpp_linear_bf16.bin")
    elif stage == 2:
        save_cpu(dyt_alpha, out_dir / "dyt_alpha.bin")
        save_cpu(dyt_gamma, out_dir / "dyt_gamma.bin")
        save_cpu(dyt_beta, out_dir / "dyt_beta.bin")
        save_cpu(dyt_out, out_dir / "dyt_out.bin")
        if dyt_out_from_cpp is not None:
            save_cpu(dyt_out_from_cpp, out_dir / "dyt_out_from_cpp_linear.bin")
    save_cpu(loss.view(1), out_dir / "loss.bin")

    loss.backward()

    save_cpu(x.grad.to(dtype), out_dir / "input_grad.bin")
    save_cpu(weight.grad.to(dtype), out_dir / "weight_grad.bin")
    save_cpu(bias.grad.to(dtype), out_dir / "bias_grad.bin")
    save_cpu(linear_out.grad.to(dtype), out_dir / "linear_out_grad.bin")
    if stage == 1:
        save_cpu(ln_gamma.grad.to(dtype), out_dir / "ln_gamma_grad.bin")
        save_cpu(ln_beta.grad.to(dtype), out_dir / "ln_beta_grad.bin")
        save_cpu(ln_out.grad.to(dtype), out_dir / "ln_out_grad.bin")
    elif stage == 2:
        save_cpu(dyt_alpha.grad.to(dtype), out_dir / "dyt_alpha_grad.bin")
        save_cpu(dyt_gamma.grad.to(dtype), out_dir / "dyt_gamma_grad.bin")
        save_cpu(dyt_beta.grad.to(dtype), out_dir / "dyt_beta_grad.bin")
        save_cpu(dyt_out.grad.to(dtype), out_dir / "dyt_out_grad.bin")

    print("=" * 70)
    print("Comparing PyTorch outputs vs C++ dumps")
    print("=" * 70)

    compare_files = [
        "input.bin",
        "target.bin",
        "weight.bin",
        "bias.bin",
        "linear_out.bin",
        "loss.bin",
        "input_grad.bin",
        "weight_grad.bin",
        "bias_grad.bin",
        "linear_out_grad.bin",
    ]
    if stage == 1:
        compare_files.extend([
            "ln_gamma.bin",
            "ln_beta.bin",
            "ln_out.bin",
            "ln_mean.bin",
            "ln_rstd.bin",
            "ln_x_norm.bin",
            "ln_x_centered.bin",
            "ln_mean_broadcast.bin",
            "ln_rstd_broadcast.bin",
            "ln_gamma_grad.bin",
            "ln_beta_grad.bin",
            "ln_out_grad.bin",
        ])
    elif stage == 2:
        compare_files.extend([
            "dyt_alpha.bin",
            "dyt_gamma.bin",
            "dyt_beta.bin",
            "dyt_out.bin",
            "dyt_alpha_grad.bin",
            "dyt_gamma_grad.bin",
            "dyt_beta_grad.bin",
            "dyt_out_grad.bin",
        ])

    results = []
    for name in compare_files:
        cpp_path = cpp_dir / name
        torch_path = out_dir / name
        if not cpp_path.exists() or not torch_path.exists():
            print(f"  {name:24s}: SKIP (missing)")
            continue
        cpp_t = load_tensor(str(cpp_path))
        torch_t = load_tensor(str(torch_path))
        if cpp_t.shape != torch_t.shape:
            print(f"  {name:24s}: SHAPE MISMATCH {list(cpp_t.shape)} vs {list(torch_t.shape)}")
            continue
        result = compare_tensors(name.replace(".bin", ""), torch_t, cpp_t)
        print_comparison(result)
        results.append(result)

    if stage == 1 and ln_out_from_cpp is not None:
        cpp_ln = load_tensor(str(cpp_dir / "ln_out.bin"))
        torch_ln_cpp = load_tensor(str(out_dir / "ln_out_from_cpp_linear.bin"))
        if cpp_ln.shape == torch_ln_cpp.shape:
            result = compare_tensors("ln_out_from_cpp_linear", torch_ln_cpp, cpp_ln)
            print_comparison(result)
        else:
            print("  ln_out_from_cpp_linear   : SHAPE MISMATCH")

    if stage == 1:
        # Compare LN stats computed in PyTorch vs TTNN manual LN buffers
        cpp_mean_path = cpp_dir / "ln_mean.bin"
        cpp_rstd_path = cpp_dir / "ln_rstd.bin"
        cpp_xnorm_path = cpp_dir / "ln_x_norm.bin"
        if cpp_mean_path.exists():
            cpp_mean = load_tensor(str(cpp_mean_path))
            torch_mean = linear_out.float().mean(dim=-1, keepdim=True).to(dtype)
            if cpp_mean.shape == torch_mean.shape:
                result = compare_tensors("ln_mean", torch_mean, cpp_mean)
                print_comparison(result)
            else:
                print("  ln_mean                  : SHAPE MISMATCH")
        if cpp_rstd_path.exists():
            cpp_rstd = load_tensor(str(cpp_rstd_path))
            var = (linear_out.float() - linear_out.float().mean(dim=-1, keepdim=True)) ** 2
            var = var.mean(dim=-1, keepdim=True)
            torch_rstd = (var + eps).rsqrt().to(dtype)
            if cpp_rstd.shape == torch_rstd.shape:
                result = compare_tensors("ln_rstd", torch_rstd, cpp_rstd)
                print_comparison(result)
            else:
                print("  ln_rstd                  : SHAPE MISMATCH")
        if cpp_xnorm_path.exists():
            cpp_xnorm = load_tensor(str(cpp_xnorm_path))
            mean = linear_out.float().mean(dim=-1, keepdim=True)
            var = ((linear_out.float() - mean) ** 2).mean(dim=-1, keepdim=True)
            torch_xnorm = ((linear_out.float() - mean) * (var + eps).rsqrt()).to(dtype)
            if cpp_xnorm.shape == torch_xnorm.shape:
                result = compare_tensors("ln_x_norm", torch_xnorm, cpp_xnorm)
                print_comparison(result)
            else:
                print("  ln_x_norm                : SHAPE MISMATCH")

        # Compare x_norm using C++ mean/rstd buffers (isolates mean/rstd vs multiply)
        if cpp_xnorm_path.exists() and cpp_mean_path.exists() and cpp_rstd_path.exists():
            cpp_xnorm = load_tensor(str(cpp_xnorm_path))
            cpp_mean = load_tensor(str(cpp_mean_path)).to(dtype)
            cpp_rstd = load_tensor(str(cpp_rstd_path)).to(dtype)
            torch_xnorm_cppstats = (linear_out.to(dtype) - cpp_mean) * cpp_rstd
            if cpp_xnorm.shape == torch_xnorm_cppstats.shape:
                result = compare_tensors("ln_x_norm_cppstats", torch_xnorm_cppstats, cpp_xnorm)
                print_comparison(result)
            else:
                print("  ln_x_norm_cppstats       : SHAPE MISMATCH")
        cpp_xc_path = cpp_dir / "ln_x_centered.bin"
        if cpp_xc_path.exists():
            cpp_xc = load_tensor(str(cpp_xc_path))
            torch_xc = (linear_out.float() - linear_out.float().mean(dim=-1, keepdim=True)).to(dtype)
            if cpp_xc.shape == torch_xc.shape:
                result = compare_tensors("ln_x_centered", torch_xc, cpp_xc)
                print_comparison(result)
            else:
                print("  ln_x_centered            : SHAPE MISMATCH")
        cpp_mean_b_path = cpp_dir / "ln_mean_broadcast.bin"
        if cpp_mean_b_path.exists():
            cpp_mean_b = load_tensor(str(cpp_mean_b_path))
            torch_mean = linear_out.float().mean(dim=-1, keepdim=True)
            torch_mean_b = torch_mean.repeat(1, 1, out_dim).to(dtype)
            if cpp_mean_b.shape == torch_mean_b.shape:
                result = compare_tensors("ln_mean_broadcast", torch_mean_b, cpp_mean_b)
                print_comparison(result)
            else:
                print("  ln_mean_broadcast        : SHAPE MISMATCH")
        cpp_rstd_b_path = cpp_dir / "ln_rstd_broadcast.bin"
        if cpp_rstd_b_path.exists():
            cpp_rstd_b = load_tensor(str(cpp_rstd_b_path))
            mean = linear_out.float().mean(dim=-1, keepdim=True)
            var = ((linear_out.float() - mean) ** 2).mean(dim=-1, keepdim=True)
            torch_rstd = (var + eps).rsqrt()
            torch_rstd_b = torch_rstd.repeat(1, 1, out_dim).to(dtype)
            if cpp_rstd_b.shape == torch_rstd_b.shape:
                result = compare_tensors("ln_rstd_broadcast", torch_rstd_b, cpp_rstd_b)
                print_comparison(result)
            else:
                print("  ln_rstd_broadcast        : SHAPE MISMATCH")

    if stage == 1 and ln_out_from_cpp_bf16 is not None:
        cpp_ln = load_tensor(str(cpp_dir / "ln_out.bin"))
        torch_ln_cpp_bf16 = load_tensor(str(out_dir / "ln_out_from_cpp_linear_bf16.bin"))
        if cpp_ln.shape == torch_ln_cpp_bf16.shape:
            result = compare_tensors("ln_out_from_cpp_linear_bf16", torch_ln_cpp_bf16, cpp_ln)
            print_comparison(result)
        else:
            print("  ln_out_from_cpp_linear_bf16: SHAPE MISMATCH")

    if stage == 1 and (cpp_dir / "ln_out_builtin.bin").exists() and ln_out_from_cpp is not None:
        cpp_builtin = load_tensor(str(cpp_dir / "ln_out_builtin.bin"))
        if cpp_builtin.shape == torch_ln_cpp.shape:
            result = compare_tensors("ln_out_builtin_vs_torch_ln", torch_ln_cpp, cpp_builtin)
            print_comparison(result)
        else:
            print("  ln_out_builtin_vs_torch_ln : SHAPE MISMATCH")

    if stage == 1 and (cpp_dir / "ln_out_builtin.bin").exists() and ln_out_from_cpp_bf16 is not None:
        cpp_builtin = load_tensor(str(cpp_dir / "ln_out_builtin.bin"))
        if cpp_builtin.shape == torch_ln_cpp_bf16.shape:
            result = compare_tensors("ln_out_builtin_vs_torch_ln_bf16", torch_ln_cpp_bf16, cpp_builtin)
            print_comparison(result)
        else:
            print("  ln_out_builtin_vs_torch_ln_bf16: SHAPE MISMATCH")

    if stage == 2 and dyt_out_from_cpp is not None:
        cpp_dyt = load_tensor(str(cpp_dir / "dyt_out.bin"))
        torch_dyt_cpp = load_tensor(str(out_dir / "dyt_out_from_cpp_linear.bin"))
        if cpp_dyt.shape == torch_dyt_cpp.shape:
            result = compare_tensors("dyt_out_from_cpp_linear", torch_dyt_cpp, cpp_dyt)
            print_comparison(result)
        else:
            print("  dyt_out_from_cpp_linear : SHAPE MISMATCH")

    if results:
        max_diff = max(r["max_diff"] for r in results)
        mean_diff = sum(r["mean_diff"] for r in results) / len(results)
        worst = max(results, key=lambda r: r["max_diff"])
        print("=" * 70)
        print(f"Worst max_diff: {worst['name']} = {worst['max_diff']:.6f}")
        print(f"Average mean_diff: {mean_diff:.6f}")
        print(f"Overall max_diff: {max_diff:.6f}")


if __name__ == "__main__":
    main()
