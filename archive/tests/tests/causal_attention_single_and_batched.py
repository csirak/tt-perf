import os
import sys
import math
import torch
import ttnn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.engine import Tensor
from src.nn import CausalAttention


def test_causal_attention_forward_backward():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(7)
        seq = 32
        dim = 32
        q = torch.randn(seq, dim)
        k = torch.randn(seq, dim)
        v = torch.randn(seq, dim)

        tq = Tensor(q, requires_grad=True, device=device)
        tk = Tensor(k, requires_grad=True, device=device)
        tv = Tensor(v, requires_grad=True, device=device)

        attn = CausalAttention()
        out = attn(tq, tk, tv)
        out_t = out.to_torch().float()

        q_b = q.to(torch.bfloat16).requires_grad_(True)
        k_b = k.to(torch.bfloat16).requires_grad_(True)
        v_b = v.to(torch.bfloat16).requires_grad_(True)
        scores = (q_b @ k_b.transpose(-2, -1)) * (1.0 / math.sqrt(dim))
        mask = torch.triu(torch.full((seq, seq), -1e9, dtype=torch.bfloat16), diagonal=1)
        scores = scores + mask
        attn_ref = torch.softmax(scores, dim=-1)
        out_ref = attn_ref @ v_b

        assert torch.allclose(out_t, out_ref.float(), atol=1e-1, rtol=1e-1)

        grad = torch.randn_like(out_t)
        out.backward(grad)
        out_ref.backward(grad.to(torch.bfloat16))

        assert torch.allclose(tq.grad_to_torch().float(), q_b.grad.float(), atol=1e-1, rtol=1e-1)
        assert torch.allclose(tk.grad_to_torch().float(), k_b.grad.float(), atol=1e-1, rtol=1e-1)
        assert torch.allclose(tv.grad_to_torch().float(), v_b.grad.float(), atol=1e-1, rtol=1e-1)
    finally:
        ttnn.close_device(device)


def test_causal_attention_batched_forward_backward():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(8)
        batch = 32
        seq = 32
        dim = 32
        q = torch.randn(batch, seq, dim)
        k = torch.randn(batch, seq, dim)
        v = torch.randn(batch, seq, dim)

        tq = Tensor(q, requires_grad=True, device=device)
        tk = Tensor(k, requires_grad=True, device=device)
        tv = Tensor(v, requires_grad=True, device=device)

        attn = CausalAttention()
        out = attn(tq, tk, tv)
        out_t = out.to_torch().float()

        q_b = q.to(torch.bfloat16).requires_grad_(True)
        k_b = k.to(torch.bfloat16).requires_grad_(True)
        v_b = v.to(torch.bfloat16).requires_grad_(True)
        scores = (q_b @ k_b.transpose(-2, -1)) * (1.0 / math.sqrt(dim))
        mask = torch.triu(torch.full((seq, seq), -1e9, dtype=torch.bfloat16), diagonal=1)
        scores = scores + mask
        attn_ref = torch.softmax(scores, dim=-1)
        out_ref = attn_ref @ v_b

        assert torch.allclose(out_t, out_ref.float(), atol=1e-1, rtol=1e-1)

        grad = torch.randn_like(out_t)
        out.backward(grad)
        out_ref.backward(grad.to(torch.bfloat16))

        assert torch.allclose(tq.grad_to_torch().float(), q_b.grad.float(), atol=1e-1, rtol=1e-1)
        assert torch.allclose(tk.grad_to_torch().float(), k_b.grad.float(), atol=1e-1, rtol=1e-1)
        assert torch.allclose(tv.grad_to_torch().float(), v_b.grad.float(), atol=1e-1, rtol=1e-1)
    finally:
        ttnn.close_device(device)
