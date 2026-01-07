import os
import sys
import torch
import ttnn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.engine import Tensor
from src.nn import Linear
from src.optim import SGD


def test_sgd_step_updates_weight():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(21)
        x = torch.randn(32, 32)
        linear = Linear(32, 32, device=device)

        w0 = linear.weight.to_torch().to(torch.bfloat16)
        tx = Tensor(x, requires_grad=True, device=device)
        y = linear(tx)
        grad = torch.ones_like(y.to_torch())
        y.backward(grad)

        x_b = x.to(torch.bfloat16)
        grad_b = grad.to(torch.bfloat16)
        grad_w_ref = x_b.transpose(-2, -1) @ grad_b

        opt = SGD(linear.parameters(), lr=0.1)
        opt.step()

        w1 = linear.weight.to_torch().to(torch.bfloat16)
        w_ref = w0 - 0.1 * grad_w_ref
        assert torch.allclose(w1.float(), w_ref.float(), atol=5e-2, rtol=5e-2)
    finally:
        ttnn.close_device(device)


def test_sgd_two_steps_matches_torch():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(22)
        x = torch.randn(32, 32)
        linear = Linear(32, 32, device=device)
        opt = SGD(linear.parameters(), lr=0.1)

        w_t = linear.weight.to_torch().to(torch.bfloat16)

        def torch_step(weight):
            x_b = x.to(torch.bfloat16)
            y = x_b @ weight
            z = torch.tanh(y)
            grad = torch.ones_like(z)
            dy = 1 - z * z
            grad_z = grad * dy
            grad_w = x_b.transpose(-2, -1) @ grad_z
            return weight - 0.1 * grad_w

        w_t1 = torch_step(w_t)
        w_t2 = torch_step(w_t1)

        tx = Tensor(x, requires_grad=True, device=device)
        y = linear(tx).tanh()
        y.backward(torch.ones_like(y.to_torch()))
        opt.step()
        opt.zero_grad()

        y = linear(tx).tanh()
        y.backward(torch.ones_like(y.to_torch()))
        opt.step()

        w_after = linear.weight.to_torch().to(torch.bfloat16)
        assert torch.allclose(w_after.float(), w_t2.float(), atol=5e-2, rtol=5e-2)
    finally:
        ttnn.close_device(device)


def test_sgd_zero_grad_clears():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(23)
        x = torch.randn(32, 32)
        linear = Linear(32, 32, device=device)
        tx = Tensor(x, requires_grad=True, device=device)
        y = linear(tx)
        y.backward(torch.ones_like(y.to_torch()))
        assert linear.weight.grad is not None
        SGD(linear.parameters(), lr=0.1).zero_grad()
        assert linear.weight.grad is None
    finally:
        ttnn.close_device(device)


def test_sgd_skips_none_grad():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(24)
        x = torch.randn(32, 32)
        linear = Linear(32, 32, device=device)
        w0 = linear.weight.to_torch().clone()
        linear.weight.grad = None
        SGD(linear.parameters(), lr=0.1).step()
        w1 = linear.weight.to_torch()
        assert torch.allclose(w0, w1)
    finally:
        ttnn.close_device(device)


def test_sgd_lr_validation():
    try:
        SGD([], lr=0.0)
        assert False, "expected ValueError for lr=0"
    except ValueError:
        pass
    try:
        SGD([], lr=-1.0)
        assert False, "expected ValueError for lr<0"
    except ValueError:
        pass


if __name__ == "__main__":
    test_sgd_step_updates_weight()
    test_sgd_two_steps_matches_torch()
    test_sgd_zero_grad_clears()
    test_sgd_skips_none_grad()
    test_sgd_lr_validation()
