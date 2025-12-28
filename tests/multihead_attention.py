import os
import sys
import math
import torch
import ttnn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.engine import Tensor
from src.nn import MultiHeadAttention


def test_multihead_attention_forward_backward():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(11)
        batch = 32
        seq = 32
        dim = 64
        heads = 2
        head_dim = dim // heads

        x = torch.randn(batch, seq, dim)
        mha = MultiHeadAttention(dim, heads, device=device)
        tx = Tensor(x, requires_grad=True, device=device)
        out = mha(tx)
        out_t = out.to_torch().float()

        wq = mha.q.weight.to_torch().to(torch.bfloat16).requires_grad_(True)
        wk = mha.k.weight.to_torch().to(torch.bfloat16).requires_grad_(True)
        wv = mha.v.weight.to_torch().to(torch.bfloat16).requires_grad_(True)
        wo = mha.o.weight.to_torch().to(torch.bfloat16).requires_grad_(True)

        x_b = x.to(torch.bfloat16).requires_grad_(True)
        q = x_b @ wq
        k = x_b @ wk
        v = x_b @ wv
        q = q.view(batch, seq, heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(batch, seq, heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(batch, seq, heads, head_dim).permute(0, 2, 1, 3)

        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        mask = torch.triu(torch.full((seq, seq), -1e9, dtype=torch.bfloat16), diagonal=1)
        scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        attn_out = attn @ v
        attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(batch, seq, dim)
        out_ref = attn_out @ wo

        assert torch.allclose(out_t, out_ref.float(), atol=8e-2, rtol=8e-2)

        grad = torch.randn_like(out_t)
        out.backward(grad)
        out_ref.backward(grad.to(torch.bfloat16))

        assert torch.allclose(tx.grad_to_torch().float(), x_b.grad.float(), atol=2.5e-1, rtol=2.5e-1)
        assert torch.allclose(mha.q.weight.grad_to_torch().float(), wq.grad.float(), atol=1.0, rtol=1.0)
    finally:
        ttnn.close_device(device)
