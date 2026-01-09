#!/usr/bin/env python3
"""PyTorch functional MNIST MLP with TTNN parity dumps."""

import argparse
import json
import math
import struct
from pathlib import Path

import torch
import numpy as np

import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.append(str(REPO_ROOT / "experiments" / "grok"))
from tensor_io import load_tensor, save_tensor


def load_kv_file(path: Path) -> dict:
    kv = {}
    if not path.exists():
        return kv
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        kv[key.strip()] = val.strip()
    return kv


def get_int(kv: dict, key: str, default: int) -> int:
    return int(kv.get(key, default))


def get_float(kv: dict, key: str, default: float) -> float:
    return float(kv.get(key, default))


def get_str(kv: dict, key: str, default: str) -> str:
    return str(kv.get(key, default))


def load_u32_tensor(path: Path) -> torch.Tensor:
    with path.open("rb") as f:
        ndim = struct.unpack("<I", f.read(4))[0]
        shape = [struct.unpack("<I", f.read(4))[0] for _ in range(ndim)]
        numel = 1
        for d in shape:
            numel *= d
        data = f.read(numel * 4)
    arr = np.frombuffer(data, dtype=np.uint32).copy()
    return torch.from_numpy(arr).reshape(shape)


def save_u32_tensor(t: torch.Tensor, path: Path) -> None:
    t = t.detach().cpu().to(torch.uint32).contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<I", t.ndim))
        for dim in t.shape:
            f.write(struct.pack("<I", dim))
        f.write(t.numpy().tobytes())


class AdamWFP32Master:
    def __init__(self, params, lr, betas, eps, weight_decay):
        self.params = [p for p, _ in params]
        self.decay_mask = [apply_wd for _, apply_wd in params]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        self.state = []
        for p in self.params:
            self.state.append({
                "m": torch.zeros_like(p, dtype=p.dtype),
                "v": torch.zeros_like(p, dtype=p.dtype),
                "master": p.detach().float().clone(),
            })

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.step_count += 1
        bc1 = 1.0 - self.beta1 ** self.step_count
        bc2 = 1.0 - self.beta2 ** self.step_count
        step_size = self.lr * math.sqrt(bc2) / bc1

        for p, apply_wd, st in zip(self.params, self.decay_mask, self.state):
            if p.grad is None:
                continue
            grad = p.grad.to(p.dtype)
            st["m"] = st["m"] * self.beta1 + grad * (1.0 - self.beta1)
            st["v"] = st["v"] * self.beta2 + (grad * grad) * (1.0 - self.beta2)

            update = st["m"] / (st["v"].sqrt() + self.eps)
            update = update * step_size

            upd_fp32 = update.float()
            if apply_wd and self.weight_decay > 0.0:
                st["master"] = st["master"] * (1.0 - self.lr * self.weight_decay) - upd_fp32
            else:
                st["master"] = st["master"] - upd_fp32

            p.data.copy_(st["master"].to(p.dtype))


def mlp_forward(x, w1, b1, w2, b2):
    h1 = torch.matmul(x, w1.t()) + b1
    relu_out = torch.relu(h1)
    logits = torch.matmul(relu_out, w2.t()) + b2
    return h1, relu_out, logits


def cross_entropy(logits, labels, num_classes, dtype):
    softmax = torch.softmax(logits, dim=-1).to(dtype)
    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).to(dtype)
    log_probs = torch.log(softmax)
    nll = (one_hot * log_probs).sum(dim=-1)
    loss = (-nll).mean().to(dtype)
    return loss, softmax


def write_meta(out_dir: Path, cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST train PyTorch functional parity")
    parser.add_argument("--config", type=str, default=str(THIS_DIR / "default.yaml"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cpp-dir", type=str, default=None,
                        help="Load initial weights from C++ outputs dir (step_0000)")
    args = parser.parse_args()

    kv = load_kv_file(Path(args.config))
    data_dir = Path(get_str(kv, "data_dir", str(THIS_DIR / "data")))
    outputs_dir = Path(get_str(kv, "outputs_dir", str(THIS_DIR / "outputs"))) / "pytorch"

    train_samples = get_int(kv, "train_samples", 256)
    seed = get_int(kv, "seed", 1)
    pad_to = get_int(kv, "pad_to", 1024)
    num_classes = get_int(kv, "num_classes", 32)
    hidden_dim = get_int(kv, "hidden_dim", 256)
    init_std = get_float(kv, "init_std", 0.02)

    batch_size = get_int(kv, "batch_size", 32)
    steps = get_int(kv, "steps", 10)
    lr = get_float(kv, "lr", 1e-3)
    beta1 = get_float(kv, "beta1", 0.9)
    beta2 = get_float(kv, "beta2", 0.999)
    eps = get_float(kv, "eps", 1e-8)
    weight_decay = get_float(kv, "weight_decay", 1e-2)
    dump_steps = get_int(kv, "dump_steps", 10)

    device = torch.device(args.device)

    x_path = data_dir / "train_images.bin"
    y_path = data_dir / "train_labels_u32.bin"

    x_all = load_tensor(str(x_path)).view(train_samples, pad_to)
    y_all = load_u32_tensor(y_path).view(train_samples).to(torch.int64)

    x_all = x_all.to(device=device, dtype=torch.bfloat16)
    y_all = y_all.to(device=device)

    torch.manual_seed(seed)

    def init_uniform(shape):
        bound = math.sqrt(3.0) * init_std
        return (torch.rand(shape, device=device, dtype=torch.bfloat16) * 2.0 - 1.0) * bound

    if args.cpp_dir:
        cpp_dir = Path(args.cpp_dir)
        step0 = cpp_dir / "step_0000"
        if step0.exists():
            w1 = load_tensor(str(step0 / "w1.bin")).to(device=device, dtype=torch.bfloat16)
            b1 = load_tensor(str(step0 / "b1.bin")).to(device=device, dtype=torch.bfloat16)
            w2 = load_tensor(str(step0 / "w2.bin")).to(device=device, dtype=torch.bfloat16)
            b2 = load_tensor(str(step0 / "b2.bin")).to(device=device, dtype=torch.bfloat16)
        else:
            raise SystemExit(f"cpp_dir provided but {step0} not found")
    else:
        w1 = init_uniform((hidden_dim, pad_to))
        b1 = torch.zeros((1, hidden_dim), device=device, dtype=torch.bfloat16)
        w2 = init_uniform((num_classes, hidden_dim))
        b2 = torch.zeros((1, num_classes), device=device, dtype=torch.bfloat16)

    w1.requires_grad_()
    b1.requires_grad_()
    w2.requires_grad_()
    b2.requires_grad_()

    opt = AdamWFP32Master(
        [(w1, True), (b1, False), (w2, True), (b2, False)],
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )

    cfg = {
        "train_samples": train_samples,
        "batch_size": batch_size,
        "steps": steps,
        "pad_to": pad_to,
        "num_classes": num_classes,
        "hidden_dim": hidden_dim,
        "init_std": init_std,
        "lr": lr,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "weight_decay": weight_decay,
    }
    write_meta(outputs_dir, cfg)

    losses_path = outputs_dir / "losses.tsv"
    losses_path.write_text("step\tloss\n", encoding="utf-8")

    for step in range(steps):
        start = (step * batch_size) % train_samples
        idx = torch.arange(start, start + batch_size, device=device) % train_samples
        x = x_all.index_select(0, idx).contiguous()
        y = y_all.index_select(0, idx).contiguous()

        opt.zero_grad()
        h1, relu_out, logits = mlp_forward(x, w1, b1, w2, b2)
        logits.retain_grad()
        loss, softmax = cross_entropy(logits, y, num_classes, dtype=torch.bfloat16)
        loss.backward()

        if step < dump_steps:
            step_dir = outputs_dir / f"step_{step:04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            save_tensor(x, step_dir / "input.bin")
            save_u32_tensor(y, step_dir / "labels_u32.bin")
            save_tensor(w1, step_dir / "w1.bin")
            save_tensor(b1, step_dir / "b1.bin")
            save_tensor(w2, step_dir / "w2.bin")
            save_tensor(b2, step_dir / "b2.bin")
            save_tensor(h1, step_dir / "linear1_out.bin")
            save_tensor(relu_out, step_dir / "relu_out.bin")
            save_tensor(logits, step_dir / "logits.bin")
            save_tensor(softmax, step_dir / "softmax.bin")
            save_tensor(loss.view(1), step_dir / "loss.bin")
            save_tensor(w1.grad, step_dir / "w1_grad.bin")
            save_tensor(b1.grad, step_dir / "b1_grad.bin")
            save_tensor(w2.grad, step_dir / "w2_grad.bin")
            save_tensor(b2.grad, step_dir / "b2_grad.bin")
            save_tensor(logits.grad, step_dir / "logits_grad.bin")

        with losses_path.open("a", encoding="utf-8") as f:
            f.write(f"{step}\t{loss.item()}\n")

        print(f"step {step} | loss {loss.item():.6f}")
        opt.step()


if __name__ == "__main__":
    main()
