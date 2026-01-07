import os
import sys
import torch
import ttnn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.engine import Tensor
from src.nn import CrossEntropyLoss


def test_cross_entropy_forward_matches_torch():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(41)
        logits = torch.randn(32, 32, 32)
        targets = torch.randint(0, 32, (32, 32))

        loss_fn = CrossEntropyLoss()
        loss = loss_fn(Tensor(logits, device=device), targets)

        ref_logits = logits.to(torch.bfloat16).float()
        ref = torch.nn.functional.cross_entropy(
            ref_logits.reshape(-1, ref_logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        )
        out = loss.to_torch().float().item()
        assert abs(out - ref.item()) < 1e-1
    finally:
        ttnn.close_device(device)


def test_cross_entropy_backward_matches_torch():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(42)
        logits = torch.randn(32, 32, 32)
        targets = torch.randint(0, 32, (32, 32))

        logits_tt = Tensor(logits, requires_grad=True, device=device)
        loss = CrossEntropyLoss()(logits_tt, targets)
        loss.backward()

        ref_logits = logits.to(torch.bfloat16).float()
        ref_probs = torch.softmax(ref_logits, dim=-1)
        ref_one_hot = torch.nn.functional.one_hot(targets, num_classes=32).float()
        ref_grad = (ref_probs - ref_one_hot) / (ref_logits.shape[0] * ref_logits.shape[1])

        grad = logits_tt.grad_to_torch().float()
        assert torch.allclose(grad, ref_grad, atol=1e-1, rtol=1e-1)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_cross_entropy_forward_matches_torch()
    test_cross_entropy_backward_matches_torch()
