import os
import sys
import math
import torch
import ttnn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.engine import Tensor
from src.nn import TransformerBlock


def _torch_block(x, weights):
    wq, wk, wv, wo, w1, w2 = weights
    q = x @ wq
    k = x @ wk
    v = x @ wv
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(x.shape[-1]))
    mask = torch.triu(torch.full((x.shape[-2], x.shape[-2]), -1e9, dtype=x.dtype), diagonal=1)
    scores = scores + mask
    attn = torch.softmax(scores, dim=-1)
    attn_out = attn @ v
    attn_out = attn_out @ wo
    y = torch.tanh(x + attn_out)
    ffn = torch.relu(y @ w1)
    ffn = ffn @ w2
    z = torch.tanh(y + ffn)
    return z


def test_transformer_block_forward_backward():
    device = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(10)
        batch = 32
        seq = 32
        dim = 32
        x = torch.randn(batch, seq, dim)

        block = TransformerBlock(dim, ffn_mult=4, device=device)
        tx = Tensor(x, requires_grad=True, device=device)
        out = block(tx)
        out_t = out.to_torch().float()

        wq = block.q.weight.to_torch().to(torch.bfloat16).requires_grad_(True)
        wk = block.k.weight.to_torch().to(torch.bfloat16).requires_grad_(True)
        wv = block.v.weight.to_torch().to(torch.bfloat16).requires_grad_(True)
        wo = block.o.weight.to_torch().to(torch.bfloat16).requires_grad_(True)
        w1 = block.ffn1.weight.to_torch().to(torch.bfloat16).requires_grad_(True)
        w2 = block.ffn2.weight.to_torch().to(torch.bfloat16).requires_grad_(True)

        x_b = x.to(torch.bfloat16).requires_grad_(True)
        out_ref = _torch_block(x_b, (wq, wk, wv, wo, w1, w2))

        assert torch.allclose(out_t, out_ref.float(), atol=8e-2, rtol=8e-2)

        grad = torch.randn_like(out_t)
        out.backward(grad)
        out_ref.backward(grad.to(torch.bfloat16))

        assert torch.allclose(tx.grad_to_torch().float(), x_b.grad.float(), atol=3e-1, rtol=3e-1)
        assert torch.allclose(block.q.weight.grad_to_torch().float(), wq.grad.float(), atol=1.0, rtol=1.0)
        assert torch.allclose(block.ffn1.weight.grad_to_torch().float(), w1.grad.float(), atol=1.0, rtol=1.0)
    finally:
        ttnn.close_device(device)
