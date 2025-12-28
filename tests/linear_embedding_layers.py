import os
import sys
import torch
import ttnn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.engine import Tensor
from src.nn import Embedding, Linear


def test_linear_forward_backward():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(3)
        x = torch.randn(32, 32)
        linear = Linear(32, 32, device=device)

        y = linear(x)
        y_t = y.to_torch().float()
        w = linear.weight.to_torch().float()
        ref = (x.to(torch.bfloat16) @ w.to(torch.bfloat16)).float()
        assert torch.allclose(y_t, ref, atol=5e-2, rtol=5e-2)

        grad = torch.ones_like(y_t)
        y.backward(grad)
        grad_b = grad.to(torch.bfloat16)
        x_b = x.to(torch.bfloat16)
        w_b = w.to(torch.bfloat16)
        grad_w_ref = x_b.transpose(-2, -1) @ grad_b
        assert torch.allclose(linear.weight.grad_to_torch().float(), grad_w_ref.float(), atol=5e-2, rtol=5e-2)
    finally:
        ttnn.close_device(device)


def test_embedding_forward_backward():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(4)
        vocab = 32
        dim = 32
        batch = 32
        indices = torch.randint(0, vocab, (batch,), dtype=torch.int64)
        print(indices.shape, indices.dtype)
        emb = Embedding(vocab, dim, device=device)

        y = emb(indices)
        y_t = y.to_torch().float()
        w = emb.weight.to_torch().float()
        one_hot = torch.nn.functional.one_hot(indices, num_classes=vocab).to(torch.float32)
        ref = (one_hot.to(torch.bfloat16) @ w.to(torch.bfloat16)).float()
        assert torch.allclose(y_t, ref, atol=5e-2, rtol=5e-2)

        grad = torch.ones_like(y_t)
        y.backward(grad)
        grad_b = grad.to(torch.bfloat16)
        one_hot_b = one_hot.to(torch.bfloat16)
        grad_w_ref = one_hot_b.transpose(-2, -1) @ grad_b
        assert torch.allclose(emb.weight.grad_to_torch().float(), grad_w_ref.float(), atol=5e-2, rtol=5e-2)
    finally:
        ttnn.close_device(device)


def test_embedding_2d_forward_backward():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(5)
        vocab = 32
        dim = 32
        batch = 32
        seq = 32
        indices = torch.randint(0, vocab, (batch, seq), dtype=torch.int64)
        emb = Embedding(vocab, dim, device=device)

        y = emb(indices)
        y_t = y.to_torch().float()
        w = emb.weight.to_torch().float()

        flat = indices.reshape(-1)
        one_hot = torch.nn.functional.one_hot(flat, num_classes=vocab).to(torch.float32)
        ref_flat = (one_hot.to(torch.bfloat16) @ w.to(torch.bfloat16)).float()
        ref = ref_flat.reshape(batch, seq, dim)
        assert torch.allclose(y_t, ref, atol=5e-2, rtol=5e-2)

        grad = torch.ones_like(y_t)
        y.backward(grad)
        grad_b = grad.reshape(-1, dim).to(torch.bfloat16)
        one_hot_b = one_hot.to(torch.bfloat16)
        grad_w_ref = one_hot_b.transpose(-2, -1) @ grad_b
        assert torch.allclose(emb.weight.grad_to_torch().float(), grad_w_ref.float(), atol=5e-2, rtol=5e-2)
    finally:
        ttnn.close_device(device)
