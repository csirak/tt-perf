#!/usr/bin/env python3
"""
Benchmark: 2-layer MLP training step (forward + backward + SGD)
Matches the C++ benchmark in bench/bench_autograd.cpp
"""

import time
import torch
import ttnn

from engine import Tensor, resolve_tt_dtype

# Benchmark configuration (matches C++)
BATCH = 1024
DIM = 512
WARMUP = 20
ITERS = 200
LR = 0.01


class MLP:
    """Simple 2-layer MLP with ReLU activation"""

    def __init__(self, in_dim, hidden_dim, out_dim, device):
        self.device = device
        dtype = torch.bfloat16

        # Initialize weights (matches C++ init of 0.01)
        self.w1 = Tensor(
            torch.full((in_dim, hidden_dim), 0.01, dtype=dtype),
            requires_grad=True,
            device=device
        )
        self.w2 = Tensor(
            torch.full((hidden_dim, out_dim), 0.01, dtype=dtype),
            requires_grad=True,
            device=device
        )

    def forward(self, x):
        h = x @ self.w1
        h = h.relu()
        return h @ self.w2

    def parameters(self):
        return [self.w1, self.w2]


def mse_loss(pred, target):
    """MSE loss: mean((pred - target)^2)

    Simple approach: manually compute forward and backward, bypass autograd
    """
    # Forward: mean((pred - target)^2)
    diff_tt = ttnn.subtract(pred.tt, target.tt)
    sq_tt = ttnn.multiply(diff_tt, diff_tt)
    numel = 1
    for d in pred.tt.shape:
        numel *= d
    scale = 1.0 / numel
    summed_tt = ttnn.sum(sq_tt, dim=None, keepdim=True)
    loss_tt = ttnn.multiply(summed_tt, scale)

    # Create output with pred as parent
    out = Tensor(loss_tt, _children=(pred,), _op="mse", requires_grad=True, device=pred.device)

    def _backward():
        if pred.requires_grad:
            # d_pred = 2 * (pred - target) / numel
            # out.grad is ones for scalar loss
            d_pred = ttnn.multiply(diff_tt, 2.0 * scale)
            pred._add_grad(d_pred)

    out._backward = _backward
    return out


def sgd_step(params, lr):
    """SGD update: param = param - lr * grad"""
    for p in params:
        if p.grad is not None:
            # p.data = p.data - lr * p.grad
            lr_tensor = ttnn.full_like(p.grad, fill_value=lr)
            scaled_grad = ttnn.mul(p.grad, lr_tensor)
            p.tt = ttnn.sub(p.tt, scaled_grad)
            p.grad = None
            p._torch_cache = None


def bench_python_dynamic(device):
    """Benchmark Python dynamic autograd"""

    # Create model
    model = MLP(DIM, DIM, DIM, device)

    # Create input and target
    x = Tensor(
        torch.ones(BATCH, DIM, dtype=torch.bfloat16),
        requires_grad=False,
        device=device
    )
    target = Tensor(
        torch.full((BATCH, DIM), 0.5, dtype=torch.bfloat16),
        requires_grad=False,
        device=device
    )

    # Warmup
    for _ in range(WARMUP):
        pred = model.forward(x)
        loss = mse_loss(pred, target)
        loss.backward()
        sgd_step(model.parameters(), LR)

    ttnn.synchronize_device(device)

    # Timed iterations
    start = time.perf_counter()
    for _ in range(ITERS):
        pred = model.forward(x)
        loss = mse_loss(pred, target)
        loss.backward()
        sgd_step(model.parameters(), LR)

    ttnn.synchronize_device(device)
    end = time.perf_counter()

    return (end - start) * 1000 / ITERS  # ms per iter


def main():
    print("Opening device...")
    device = ttnn.open_device(device_id=0)

    print(f"\nPython Autograd Benchmark: 2-layer MLP")
    print("=" * 40)
    print(f"Config: batch={BATCH}, dim={DIM}, warmup={WARMUP}, iters={ITERS}\n")

    print("Running benchmark...")
    t_python = bench_python_dynamic(device)

    print(f"\nResults:")
    print(f"  python dynamic: {t_python:.3f} ms/iter")

    # Compare with C++ results if available
    print(f"\nFor comparison, C++ results were:")
    print(f"  dynamic:         0.984 ms/iter (1.00x)")
    print(f"  static+trace:    0.563 ms/iter (1.75x)")

    if t_python > 0:
        print(f"\nPython overhead vs C++ dynamic: {t_python / 0.984:.2f}x slower")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
