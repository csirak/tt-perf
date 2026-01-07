import os
import sys
import torch
import ttnn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.engine import Tensor


def test_relu_forward_backward():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(9)
        x = torch.randn(32, 32)
        tx = Tensor(x, requires_grad=True, device=device)
        ty = tx.relu()

        y = ty.to_torch().float()
        y_ref = torch.relu(x.to(torch.bfloat16)).float()
        assert torch.allclose(y, y_ref, atol=5e-2, rtol=5e-2)

        grad = torch.randn_like(y)
        ty.backward(grad)
        grad_b = grad.to(torch.bfloat16)
        mask = (x.to(torch.bfloat16) > 0).to(torch.bfloat16)
        gx_ref = grad_b * mask
        assert torch.allclose(tx.grad_to_torch().float(), gx_ref.float(), atol=5e-2, rtol=5e-2)
    finally:
        ttnn.close_device(device)
