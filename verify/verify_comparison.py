#!/usr/bin/env python3
"""
MLP Verification Script - Compare PyTorch (fp32/bf16) with TTNN C++ autograd
Configuration matches train_mlp_traced.cpp exactly.

Run: python verify_comparison.py
"""

import torch
import torch.nn as nn


def create_model(in_features, hidden, out_features, init_val):
    """Create 2-layer MLP with constant weight init."""
    model = nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_features)
    )

    # Initialize weights to constant, biases to 0
    with torch.no_grad():
        model[0].weight.fill_(init_val)
        model[0].bias.fill_(0.0)
        model[2].weight.fill_(init_val)
        model[2].bias.fill_(0.0)

    return model


def run_training(model, x, target, lr, num_steps, dtype_name):
    """Run training and print detailed values at each step."""

    # MSE loss doesn't support bf16 on CPU, so compute in fp32
    def mse_loss(pred, target):
        return nn.functional.mse_loss(pred.float(), target.float())

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    print(f"\n{'='*60}")
    print(f"Running with dtype: {dtype_name}")
    print(f"{'='*60}")

    # Initial forward (before any training)
    with torch.no_grad():
        pred = model(x)
        initial_loss = mse_loss(pred, target)

    print(f"\nInitial state (before training):")
    print(f"  Output mean: {pred.float().mean().item():.6f}")
    print(f"  Output[0,0]: {pred.float()[0,0].item():.6f}")
    print(f"  Initial loss: {initial_loss.float().item():.6f}")
    print(f"  W1[0,0]: {model[0].weight.float()[0,0].item():.6f}")
    print(f"  W2[0,0]: {model[2].weight.float()[0,0].item():.6f}")

    print(f"\nTraining steps:")
    print("-" * 60)

    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward
        pred = model(x)
        loss = mse_loss(pred, target)

        # Backward
        loss.backward()

        # Print gradients at step 0
        if step == 0:
            print(f"\nGradients at step 0:")
            print(f"  dW1[0,0]: {model[0].weight.grad.float()[0,0].item():.6f}")
            print(f"  dW1 mean: {model[0].weight.grad.float().mean().item():.6f}")
            print(f"  dW2[0,0]: {model[2].weight.grad.float()[0,0].item():.6f}")
            print(f"  dW2 mean: {model[2].weight.grad.float().mean().item():.6f}")
            print(f"  db1[0]: {model[0].bias.grad.float()[0].item():.6f}")
            print(f"  db2[0]: {model[2].bias.grad.float()[0].item():.6f}")
            print()

        # Optimizer step
        optimizer.step()

        # Print loss after update
        with torch.no_grad():
            pred_new = model(x)
            new_loss = mse_loss(pred_new, target)

        losses.append(new_loss.float().item())
        print(f"Step {step}: loss_before={loss.float().item():.6f}, loss_after={new_loss.float().item():.6f}")

    print(f"\nFinal state:")
    print(f"  W1[0,0]: {model[0].weight.float()[0,0].item():.6f}")
    print(f"  W2[0,0]: {model[2].weight.float()[0,0].item():.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")

    return initial_loss.float().item(), losses


def main():
    # Configuration - MUST match C++ train_mlp_traced.cpp exactly
    batch_size = 1024
    in_features = 512
    hidden = 256
    out_features = 128
    lr = 0.01
    init_val = 0.01
    num_steps = 5

    print("MLP Verification (PyTorch Reference)")
    print("=" * 60)
    print(f"Config:")
    print(f"  batch_size: {batch_size}")
    print(f"  in_features: {in_features}")
    print(f"  hidden_features: {hidden}")
    print(f"  out_features: {out_features}")
    print(f"  learning_rate: {lr}")
    print(f"  init_val: {init_val}")
    print(f"  num_steps: {num_steps}")

    # Create deterministic inputs
    x_fp32 = torch.ones(batch_size, in_features)
    target_fp32 = torch.full((batch_size, out_features), 0.5)

    # =========================================================================
    # FP32 (ground truth)
    # =========================================================================
    model_fp32 = create_model(in_features, hidden, out_features, init_val).float()
    initial_fp32, losses_fp32 = run_training(
        model_fp32, x_fp32, target_fp32, lr, num_steps, "float32"
    )

    # =========================================================================
    # BF16 (matches TTNN)
    # =========================================================================
    x_bf16 = x_fp32.bfloat16()
    target_bf16 = target_fp32.bfloat16()
    model_bf16 = create_model(in_features, hidden, out_features, init_val).bfloat16()
    initial_bf16, losses_bf16 = run_training(
        model_bf16, x_bf16, target_bf16, lr, num_steps, "bfloat16"
    )

    # =========================================================================
    # Summary Comparison
    # =========================================================================
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"\n{'Step':<6} {'FP32 Loss':<15} {'BF16 Loss':<15} {'Diff':<15}")
    print("-" * 60)
    print(f"{'Init':<6} {initial_fp32:<15.6f} {initial_bf16:<15.6f} {abs(initial_fp32-initial_bf16):<15.6f}")
    for i in range(num_steps):
        diff = abs(losses_fp32[i] - losses_bf16[i])
        print(f"{i:<6} {losses_fp32[i]:<15.6f} {losses_bf16[i]:<15.6f} {diff:<15.6f}")

    print(f"\n{'='*60}")
    print("EXPECTED VALUES FOR C++ COMPARISON")
    print(f"{'='*60}")
    print(f"Initial loss (bf16): {initial_bf16:.6f}")
    print(f"Final loss after 5 steps (bf16): {losses_bf16[-1]:.6f}")
    print(f"\nIf C++ autograd matches these values (within ~1% tolerance),")
    print("then the implementation is correct.")


if __name__ == "__main__":
    main()
