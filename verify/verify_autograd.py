#!/usr/bin/env python3
"""
PyTorch reference implementation to verify C++ autograd expected values.
Run: python verify_autograd.py
"""

import torch
import torch.nn.functional as F


def test_matmul():
    """Test matmul forward + backward with ones."""
    print("=" * 50)
    print("Test: Matmul")

    # A[32,64] @ B[64,32] = C[32,32], all ones
    a = torch.ones(32, 64, requires_grad=True)
    b = torch.ones(64, 32, requires_grad=True)
    c = a @ b

    # Forward: each element = 64
    print(f"Forward: C[0,0] = {c[0,0].item():.1f} (expected: 64)")
    assert torch.allclose(c, torch.full_like(c, 64.0))

    # Backward with dC = ones
    c.backward(torch.ones_like(c))

    # dA = dC @ B.T = ones(32,32) @ ones(32,64) = 32 * ones(32,64)
    print(f"dA[0,0] = {a.grad[0,0].item():.1f} (expected: 32)")
    assert torch.allclose(a.grad, torch.full_like(a.grad, 32.0))

    # dB = A.T @ dC = ones(64,32) @ ones(32,32) = 32 * ones(64,32)
    print(f"dB[0,0] = {b.grad[0,0].item():.1f} (expected: 32)")
    assert torch.allclose(b.grad, torch.full_like(b.grad, 32.0))

    print("PASSED\n")


def test_relu():
    """Test ReLU forward + backward with positive values."""
    print("=" * 50)
    print("Test: ReLU")

    x = torch.ones(32, 32, requires_grad=True)
    y = F.relu(x)

    # Forward: relu(1) = 1
    print(f"Forward: y[0,0] = {y[0,0].item():.1f} (expected: 1)")
    assert torch.allclose(y, torch.ones_like(y))

    # Backward with dy = ones
    y.backward(torch.ones_like(y))

    # For positive x: dx = dy * 1 = 1
    print(f"dx[0,0] = {x.grad[0,0].item():.1f} (expected: 1)")
    assert torch.allclose(x.grad, torch.ones_like(x.grad))

    print("PASSED\n")


def test_softmax():
    """Test softmax forward + backward with uniform input."""
    print("=" * 50)
    print("Test: Softmax")

    x = torch.ones(32, 64, requires_grad=True)
    y = F.softmax(x, dim=-1)

    # Forward: softmax(ones) = uniform = 1/64
    expected = 1.0 / 64.0
    print(f"Forward: y[0,0] = {y[0,0].item():.6f} (expected: {expected:.6f})")
    assert torch.allclose(y, torch.full_like(y, expected))

    # Backward with dy = ones
    y.backward(torch.ones_like(y))

    # For uniform softmax with uniform gradient:
    # dx = y * (dy - sum(dy * y)) = (1/64) * (1 - 1) = 0
    print(f"dx[0,0] = {x.grad[0,0].item():.6f} (expected: 0)")
    assert torch.allclose(x.grad, torch.zeros_like(x.grad), atol=1e-6)

    print("PASSED\n")


def test_linear():
    """Test Linear layer forward + backward."""
    print("=" * 50)
    print("Test: Linear")

    # x[32, 64], Linear(64 -> 32)
    x = torch.ones(32, 64, requires_grad=True)

    # Use ones for weight to make math predictable
    linear = torch.nn.Linear(64, 32, bias=True)
    with torch.no_grad():
        linear.weight.fill_(0.1)  # [32, 64]
        linear.bias.fill_(0.0)   # [32]

    y = linear(x)  # y = x @ W.T + b

    # Forward: each output = sum(0.1 * 1 * 64) + 0 = 6.4
    print(f"Forward: y[0,0] = {y[0,0].item():.1f} (expected: 6.4)")
    assert torch.allclose(y, torch.full_like(y, 6.4), atol=0.1)

    # Backward with dy = ones
    y.backward(torch.ones_like(y))

    # dBias = sum(dy, dim=0) = 32 (batch size)
    print(f"dBias[0] = {linear.bias.grad[0].item():.1f} (expected: 32)")
    assert torch.allclose(linear.bias.grad, torch.full_like(linear.bias.grad, 32.0))

    # dx shape matches input
    print(f"dx shape: {x.grad.shape} (expected: [32, 64])")
    assert x.grad.shape == (32, 64)

    print("PASSED\n")


def test_sgd():
    """Test SGD optimizer step."""
    print("=" * 50)
    print("Test: SGD")

    # Create Linear with zeros bias
    linear = torch.nn.Linear(64, 32, bias=True)
    with torch.no_grad():
        linear.weight.fill_(0.1)
        linear.bias.fill_(0.0)

    optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

    # Forward + backward
    x = torch.ones(32, 64)
    y = linear(x)
    y.backward(torch.ones_like(y))

    # Gradient check
    print(f"Bias grad[0] = {linear.bias.grad[0].item():.1f} (expected: 32)")
    assert torch.allclose(linear.bias.grad, torch.full_like(linear.bias.grad, 32.0))

    # SGD step
    optimizer.step()

    # bias = 0 - 0.01 * 32 = -0.32
    print(f"Bias after step 1: {linear.bias[0].item():.2f} (expected: -0.32)")
    assert torch.allclose(linear.bias, torch.full_like(linear.bias, -0.32), atol=0.01)

    # Second iteration
    optimizer.zero_grad()
    y2 = linear(x)
    y2.backward(torch.ones_like(y2))

    print(f"Bias grad after 2nd backward: {linear.bias.grad[0].item():.1f} (expected: 32)")
    assert torch.allclose(linear.bias.grad, torch.full_like(linear.bias.grad, 32.0))

    optimizer.step()

    # bias = -0.32 - 0.01 * 32 = -0.64
    print(f"Bias after step 2: {linear.bias[0].item():.2f} (expected: -0.64)")
    assert torch.allclose(linear.bias, torch.full_like(linear.bias, -0.64), atol=0.01)

    print("PASSED\n")


if __name__ == "__main__":
    print("\nPyTorch Reference Tests for C++ Autograd\n")

    test_matmul()
    test_relu()
    test_softmax()
    test_linear()
    test_sgd()

    print("=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)
