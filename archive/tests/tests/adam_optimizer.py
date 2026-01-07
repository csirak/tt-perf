import os
import sys
import torch
import ttnn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.engine import Tensor
from src.nn import Linear
from src.optim import Adam


def _adam_step_ref(weight, grad, lr, betas, eps, step, m, v):
    beta1, beta2 = betas
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad * grad)
    m_hat = m / (1.0 - (beta1 ** step))
    v_hat = v / (1.0 - (beta2 ** step))
    weight = weight - lr * m_hat / (torch.sqrt(v_hat) + eps)
    return weight, m, v


def test_adam_step_matches_torch():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(31)
        x = torch.randn(32, 32)
        linear = Linear(32, 32, device=device)

        w0 = linear.weight.to_torch().to(torch.float32)
        tx = Tensor(x, requires_grad=True, device=device)
        y = linear(tx)
        grad = torch.ones_like(y.to_torch())
        y.backward(grad)

        x_b = x.to(torch.float32)
        grad_b = grad.to(torch.float32)
        grad_w_ref = x_b.transpose(-2, -1) @ grad_b

        opt = Adam(linear.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-8)
        opt.step()

        w1 = linear.weight.to_torch().to(torch.float32)
        m0 = torch.zeros_like(grad_w_ref)
        v0 = torch.zeros_like(grad_w_ref)
        w_ref, _, _ = _adam_step_ref(w0, grad_w_ref, 0.1, (0.9, 0.999), 1e-8, 1, m0, v0)
        assert torch.allclose(w1, w_ref, atol=1e-1, rtol=1e-1)
    finally:
        ttnn.close_device(device)


def test_adam_two_steps_matches_torch():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(32)
        x = torch.randn(32, 32)
        linear = Linear(32, 32, device=device)
        opt = Adam(linear.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-8)

        w_t = linear.weight.to_torch().to(torch.float32)

        def torch_step(weight, step, m, v):
            x_f = x.to(torch.float32)
            y = x_f @ weight
            z = torch.tanh(y)
            grad = torch.ones_like(z)
            dy = 1 - z * z
            grad_z = grad * dy
            grad_w = x_f.transpose(-2, -1) @ grad_z
            return _adam_step_ref(weight, grad_w, 0.1, (0.9, 0.999), 1e-8, step, m, v)

        m = torch.zeros_like(w_t)
        v = torch.zeros_like(w_t)
        w_t1, m, v = torch_step(w_t, 1, m, v)
        w_t2, _, _ = torch_step(w_t1, 2, m, v)

        tx = Tensor(x, requires_grad=True, device=device)
        y = linear(tx).tanh()
        y.backward(torch.ones_like(y.to_torch()))
        opt.step()
        opt.zero_grad()

        y = linear(tx).tanh()
        y.backward(torch.ones_like(y.to_torch()))
        opt.step()

        w_after = linear.weight.to_torch().to(torch.float32)
        assert torch.allclose(w_after, w_t2, atol=1e-1, rtol=1e-1)
    finally:
        ttnn.close_device(device)


def test_adam_zero_grad_clears():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(33)
        x = torch.randn(32, 32)
        linear = Linear(32, 32, device=device)
        tx = Tensor(x, requires_grad=True, device=device)
        y = linear(tx)
        y.backward(torch.ones_like(y.to_torch()))
        assert linear.weight.grad is not None
        Adam(linear.parameters(), lr=0.1).zero_grad()
        assert linear.weight.grad is None
    finally:
        ttnn.close_device(device)


def test_adam_skips_none_grad():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(34)
        x = torch.randn(32, 32)
        linear = Linear(32, 32, device=device)
        w0 = linear.weight.to_torch().clone()
        linear.weight.grad = None
        Adam(linear.parameters(), lr=0.1).step()
        w1 = linear.weight.to_torch()
        assert torch.allclose(w0, w1)
    finally:
        ttnn.close_device(device)


def test_adam_validation():
    try:
        Adam([], lr=0.0)
        assert False, "expected ValueError for lr=0"
    except ValueError:
        pass
    try:
        Adam([], lr=-1.0)
        assert False, "expected ValueError for lr<0"
    except ValueError:
        pass
    try:
        Adam([], eps=0.0)
        assert False, "expected ValueError for eps=0"
    except ValueError:
        pass
    try:
        Adam([], betas=(0.9,))
        assert False, "expected ValueError for betas length"
    except ValueError:
        pass
    try:
        Adam([], betas=(1.1, 0.9))
        assert False, "expected ValueError for betas range"
    except ValueError:
        pass


if __name__ == "__main__":
    test_adam_step_matches_torch()
    test_adam_two_steps_matches_torch()
    test_adam_zero_grad_clears()
    test_adam_skips_none_grad()
    test_adam_validation()
