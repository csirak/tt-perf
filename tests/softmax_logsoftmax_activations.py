import os
import sys
import torch
import ttnn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.engine import Tensor


def test_softmax_forward_backward():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(5)
        x = torch.randn(32, 32)
        tx = Tensor(x, requires_grad=True, device=device)
        ty = tx.softmax(dim=-1)

        y = ty.to_torch().float()
        y_ref = torch.softmax(x.to(torch.bfloat16), dim=-1).float()
        assert torch.allclose(y, y_ref, atol=5e-2, rtol=5e-2)

        grad = torch.randn_like(y)
        ty.backward(grad)
        grad_b = grad.to(torch.bfloat16)
        y_b = y_ref.to(torch.bfloat16)
        dot = (grad_b * y_b).sum(dim=-1, keepdim=True)
        gx_ref = (grad_b - dot) * y_b
        assert torch.allclose(tx.grad_to_torch().float(), gx_ref.float(), atol=5e-2, rtol=5e-2)
    finally:
        ttnn.close_device(device)


def test_logsoftmax_forward_backward():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(6)
        x = torch.randn(32, 32)
        tx = Tensor(x, requires_grad=True, device=device)
        ty = tx.logsoftmax(dim=-1)

        y = ty.to_torch().float()
        y_ref = torch.log_softmax(x.to(torch.bfloat16), dim=-1).float()
        assert torch.allclose(y, y_ref, atol=5e-2, rtol=5e-2)

        grad = torch.randn_like(y)
        ty.backward(grad)
        grad_b = grad.to(torch.bfloat16)
        y_b = y_ref.to(torch.bfloat16)
        soft = torch.exp(y_b)
        sum_grad = grad_b.sum(dim=-1, keepdim=True)
        gx_ref = grad_b - soft * sum_grad
        assert torch.allclose(tx.grad_to_torch().float(), gx_ref.float(), atol=5e-2, rtol=5e-2)
    finally:
        ttnn.close_device(device)
