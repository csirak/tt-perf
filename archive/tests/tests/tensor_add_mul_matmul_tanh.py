import os
import sys
import torch
import ttnn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.engine import Tensor


def test_add_mul_forward_backward():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(0)
        x = torch.randn(32, 32)
        y = torch.randn(32, 32)

        tx = Tensor(x, requires_grad=True, device=device)
        ty = Tensor(y, requires_grad=True, device=device)
        tz = tx * ty + ty

        z = tz.to_torch().float()
        z_ref = x * y + y
        assert torch.allclose(z, z_ref, atol=5e-2, rtol=5e-2)

        grad = torch.ones_like(z)
        tz.backward(grad)
        gx_ref = grad * y
        gy_ref = grad * x + grad
        assert torch.allclose(tx.grad_to_torch().float(), gx_ref, atol=5e-2, rtol=5e-2)
        assert torch.allclose(ty.grad_to_torch().float(), gy_ref, atol=5e-2, rtol=5e-2)
    finally:
        ttnn.close_device(device)


def test_matmul_tanh_forward_backward():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(1)
        x = torch.randn(32, 32)
        y = torch.randn(32, 32)

        tx = Tensor(x, requires_grad=True, device=device)
        ty = Tensor(y, requires_grad=True, device=device)
        tmat = tx @ ty
        mat = tmat.to_torch().float()
        x_b = x.to(torch.bfloat16)
        y_b = y.to(torch.bfloat16)
        mat_ref = (x_b @ y_b).float()
        assert torch.allclose(mat, mat_ref, atol=5e-2, rtol=5e-2)

        tz = tmat.tanh()
        z = tz.to_torch().float()
        z_ref_b = torch.tanh(x_b @ y_b)

        grad = torch.ones_like(z)
        tz.backward(grad)
        grad_b = grad.to(torch.bfloat16)
        dy_b = 1 - z_ref_b * z_ref_b
        grad_z_b = grad_b * dy_b
        gx_ref_b = grad_z_b @ y_b.transpose(-2, -1)
        gy_ref_b = x_b.transpose(-2, -1) @ grad_z_b
        assert torch.allclose(tx.grad_to_torch().float(), gx_ref_b.float(), atol=1e-1, rtol=1e-1)
        assert torch.allclose(ty.grad_to_torch().float(), gy_ref_b.float(), atol=7.5e-2, rtol=7.5e-2)
    finally:
        ttnn.close_device(device)


def test_mixed_inputs_coerce_to_tensor():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(2)
        x = torch.randn(32, 32)
        y = torch.randn(32, 32)

        tx = Tensor(x, requires_grad=True, device=device)
        tz = tx + y
        z = tz.to_torch().float()
        ref = (x.to(torch.bfloat16) + y.to(torch.bfloat16)).float()
        assert torch.allclose(z, ref, atol=5e-2, rtol=5e-2)

        grad = torch.ones_like(z)
        tz.backward(grad)
        gx_ref = torch.ones_like(x).to(torch.bfloat16).float()
        assert torch.allclose(tx.grad_to_torch().float(), gx_ref, atol=5e-2, rtol=5e-2)

        tmat = tx @ y
        mat = tmat.to_torch().float()
        mat_ref = (x.to(torch.bfloat16) @ y.to(torch.bfloat16)).float()
        assert torch.allclose(mat, mat_ref, atol=5e-2, rtol=5e-2)
    finally:
        ttnn.close_device(device)
