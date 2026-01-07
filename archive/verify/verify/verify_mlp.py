#!/usr/bin/env python3
"""
MLP Verification Script (PyTorch)
Uses deterministic inputs to verify correctness against TTNN C++ implementation.

All values are chosen to match exactly with the C++ verify_mlp_traced.cpp.
"""

import torch
import torch.nn as nn

def main():
    # Configuration - must match C++ exactly
    # Note: dimensions must be multiples of 32 for TTNN tile alignment
    batch_size = 32
    in_features = 64
    hidden = 32
    out_features = 32
    lr = 0.1
    num_steps = 10

    print("MLP Verification (PyTorch Reference)")
    print("=" * 50)
    print(f"Config: batch={batch_size}, in={in_features}, hidden={hidden}, out={out_features}")
    print(f"Learning rate: {lr}")
    print()

    # Deterministic inputs - all ones
    x = torch.ones(batch_size, in_features)
    target = torch.full((batch_size, out_features), 0.5)

    # Create model with deterministic weights
    model = nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_features)
    )

    # Initialize weights to constant 0.01, biases to 0
    with torch.no_grad():
        model[0].weight.fill_(0.01)  # W1: [hidden, in_features]
        model[0].bias.fill_(0.0)      # b1: [hidden]
        model[2].weight.fill_(0.01)  # W2: [out_features, hidden]
        model[2].bias.fill_(0.0)      # b2: [out_features]

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Compute initial loss (before any training)
    with torch.no_grad():
        pred = model(x)
        initial_loss = criterion(pred, target)
    print(f"Initial loss (step 0): {initial_loss.item():.6f}")
    print(f"  W1[0,0] = {model[0].weight[0,0].item():.6f}")
    print(f"  W2[0,0] = {model[2].weight[0,0].item():.6f}")
    print()

    # Training loop
    print("Training steps:")
    print("-" * 50)
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        # Forward
        pred = model(x)
        loss = criterion(pred, target)

        # Backward
        loss.backward()

        # Print gradients at step 1 for debugging
        if step == 1:
            print(f"  Gradients at step 1:")
            print(f"    dW1[0,0] = {model[0].weight.grad[0,0].item():.6f}")
            print(f"    dW2[0,0] = {model[2].weight.grad[0,0].item():.6f}")
            print(f"    db1[0] = {model[0].bias.grad[0].item():.6f}")
            print(f"    db2[0] = {model[2].bias.grad[0].item():.6f}")

        # Optimizer step
        optimizer.step()

        # Print loss after update
        with torch.no_grad():
            pred = model(x)
            new_loss = criterion(pred, target)

        print(f"Step {step}: loss = {new_loss.item():.6f}")
        print(f"  W1[0,0] = {model[0].weight[0,0].item():.6f}")
        print(f"  W2[0,0] = {model[2].weight[0,0].item():.6f}")

    print()
    print("=" * 50)
    print("Summary:")
    print(f"  Initial loss: {initial_loss.item():.6f}")
    print(f"  Final loss:   {new_loss.item():.6f}")
    print(f"  Loss reduced: {'YES' if new_loss < initial_loss else 'NO'}")


if __name__ == "__main__":
    main()
