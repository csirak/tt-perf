import math
import torch
import ttnn

from src.engine import Tensor, require_same_device, resolve_tt_dtype, resolve_torch_dtype


class Parameter(Tensor):
    def __init__(self, data, *, dtype=None, device=None):
        super().__init__(data, requires_grad=True, dtype=dtype, device=device)


def _init_weight(weight, *, dtype, device, name: str, init_shape):
    if isinstance(weight, Tensor):
        if device is not None and weight.device != device:
            raise ValueError(f"device mismatch for {name} weight")
        return weight if isinstance(weight, Parameter) else Parameter(weight.tt, dtype=dtype, device=weight.device)
    if device is None:
        raise ValueError(f"device is required when initializing {name} without a Tensor weight")
    if weight is None:
        scale = 1.0 / math.sqrt(init_shape[0])
        weight = torch.randn(*init_shape, dtype=dtype) * scale
    elif isinstance(weight, torch.Tensor):
        weight = weight.to(dtype)
    return Parameter(weight, dtype=dtype, device=device)


class Linear:
    def __init__(self, in_features: int, out_features: int, weight=None, dtype=None, device=None):
        self.dtype = resolve_torch_dtype(dtype)
        self.weight = _init_weight(
            weight,
            dtype=self.dtype,
            device=device,
            name="Linear",
            init_shape=(in_features, out_features),
        )

    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, dtype=self.dtype, device=self.weight.device, compute_kernel_config=self.weight.compute_kernel_config)
        else:
            require_same_device(x, self.weight, name="Linear")
        return x @ self.weight

    def parameters(self):
        return [self.weight]


class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int, weight=None, dtype=None, device=None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = resolve_torch_dtype(dtype)
        self.weight = _init_weight(
            weight,
            dtype=self.dtype,
            device=device,
            name="Embedding",
            init_shape=(num_embeddings, embedding_dim),
        )

    def __call__(self, indices):
        if isinstance(indices, Tensor):
            require_same_device(indices, self.weight, name="Embedding")
            indices_t = indices.to_torch().to(torch.int64)
            indices_tt = None
            if isinstance(indices.tt, ttnn.Tensor):
                try:
                    if indices.tt.dtype == ttnn.uint32 and indices.tt.get_layout() == ttnn.ROW_MAJOR_LAYOUT:
                        indices_tt = indices.tt
                except Exception:
                    indices_tt = None
        elif isinstance(indices, torch.Tensor):
            indices_t = indices.to(torch.int64)
            indices_tt = None
        else:
            indices_t = torch.tensor(indices, dtype=torch.int64)
            indices_tt = None

        if indices_t.ndim not in (1, 2):
            raise ValueError("Embedding expects 1D or 2D indices")
        if indices_t.ndim == 1:
            flat_indices = indices_t
            out_shape = (indices_t.shape[0], self.embedding_dim)
            if (flat_indices.shape[0] % 32) != 0:
                raise ValueError("batch must be multiple of 32 for TILE_LAYOUT")
        else:
            batch, seq = indices_t.shape
            if (batch % 32) != 0 or (seq % 32) != 0:
                raise ValueError("batch and seq must be multiples of 32 for TILE_LAYOUT")
            flat_indices = indices_t.reshape(-1)
            out_shape = (batch, seq, self.embedding_dim)
        if (self.num_embeddings % 32) != 0 or (self.embedding_dim % 32) != 0:
            raise ValueError("num_embeddings and embedding_dim must be multiples of 32")

        if indices_tt is not None and indices_t.ndim == 2:
            indices_tt = ttnn.reshape(indices_tt, (flat_indices.shape[0],))
        if indices_tt is None:
            indices_tt = ttnn.from_torch(
                flat_indices,
                device=self.weight.device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
            )
        out_tt = ttnn.embedding(
            indices_tt,
            weight=self.weight.tt,
            dtype=resolve_tt_dtype(self.dtype),
            layout=ttnn.TILE_LAYOUT,
        )
        if len(out_shape) == 3:
            out_tt = ttnn.reshape(out_tt, out_shape)
        out = Tensor(
            out_tt,
            (self.weight,),
            "embedding",
            self.weight.requires_grad,
            compute_kernel_config=self.weight.compute_kernel_config,
        )

        def _backward():
            if self.weight.requires_grad:
                one_hot = torch.nn.functional.one_hot(flat_indices, num_classes=self.num_embeddings).to(self.dtype)
                grad_w = Tensor(
                    one_hot,
                    dtype=self.dtype,
                    device=self.weight.device,
                    compute_kernel_config=self.weight.compute_kernel_config,
                ).tt
                grad_out = out.grad
                if len(out_shape) == 3:
                    grad_out = ttnn.reshape(out.grad, (flat_indices.shape[0], self.embedding_dim))
                self.weight._add_grad(ttnn.matmul(grad_w, grad_out, transpose_a=True))

        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight]


class CrossEntropyLoss:
    def __init__(self, *, reduction: str = "mean"):
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def __call__(self, logits: Tensor, targets):
        if not isinstance(logits, Tensor):
            raise ValueError("CrossEntropyLoss expects Tensor logits")
        if logits.compute_kernel_config is None:
            raise ValueError("CrossEntropyLoss requires compute_kernel_config")

        if isinstance(targets, Tensor):
            require_same_device(logits, targets, name="CrossEntropyLoss")
            targets_t = targets.to_torch().to(torch.int64)
        elif isinstance(targets, torch.Tensor):
            targets_t = targets.to(torch.int64)
        else:
            targets_t = torch.tensor(targets, dtype=torch.int64)

        if len(logits.tt.shape) != 3:
            raise ValueError("CrossEntropyLoss expects 3D (B, S, C) logits")

        batch, seq, classes = logits.tt.shape
        if targets_t.ndim != 2:
            raise ValueError("targets must be 2D for 3D logits")
        flat_targets = targets_t.reshape(-1)
        flat_logits = ttnn.reshape(logits.tt, (batch * seq, classes))
        out_shape = (batch, seq)

        if (classes % 32) != 0:
            raise ValueError("classes must be a multiple of 32 for TILE_LAYOUT")
        if (flat_logits.shape[0] % 32) != 0:
            raise ValueError("batch*seq must be a multiple of 32 for TILE_LAYOUT")
        if flat_targets.numel() != flat_logits.shape[0]:
            raise ValueError("targets size must match logits batch*seq")

        one_hot = torch.nn.functional.one_hot(flat_targets, num_classes=classes).to(
            resolve_torch_dtype(logits.tt.dtype)
        )
        one_hot_tt = ttnn.from_torch(
            one_hot,
            device=logits.device,
            layout=ttnn.TILE_LAYOUT,
            dtype=resolve_tt_dtype(logits.tt.dtype),
        )

        softmax_tt = ttnn.softmax(flat_logits, dim=-1, compute_kernel_config=logits.compute_kernel_config)
        log_probs_tt = ttnn.log(softmax_tt)
        nll_tt = ttnn.sum(ttnn.mul(one_hot_tt, log_probs_tt), dim=-1, keepdim=False)
        nll_tt = ttnn.mul(nll_tt, ttnn.full_like(nll_tt, fill_value=-1.0))

        if self.reduction == "sum":
            loss_tt = ttnn.sum(nll_tt, dim=0, keepdim=False)
        else:
            total = ttnn.sum(nll_tt, dim=0, keepdim=False)
            scale_tt = ttnn.full_like(total, fill_value=1.0 / float(flat_logits.shape[0]))
            loss_tt = ttnn.mul(total, scale_tt)

        out = Tensor(
            loss_tt,
            (logits,),
            "cross_entropy",
            logits.requires_grad,
            compute_kernel_config=logits.compute_kernel_config,
        )

        def _backward():
            if not logits.requires_grad:
                return
            grad_scale = ttnn.to_torch(out.grad).item()
            if self.reduction == "mean":
                grad_scale /= float(flat_logits.shape[0])
            scale_tt = ttnn.full_like(softmax_tt, fill_value=grad_scale)
            grad_logits = ttnn.mul(ttnn.sub(softmax_tt, one_hot_tt), scale_tt)
            grad_logits = ttnn.reshape(grad_logits, (batch, seq, classes))
            logits._add_grad(grad_logits)

        out._backward = _backward
        return out


class CausalAttention:
    def __init__(self, *, scale: float = None):
        self.scale = scale

    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        require_same_device(q, k, v, name="CausalAttention")
        if len(q.tt.shape) not in (2, 3, 4) or len(k.tt.shape) != len(q.tt.shape) or len(v.tt.shape) != len(q.tt.shape):
            raise ValueError("CausalAttention expects 2D (S, D), 3D (B, S, D), or 4D (B, H, S, D) tensors")
        if q.tt.shape != k.tt.shape or q.tt.shape != v.tt.shape:
            raise ValueError("shape mismatch in CausalAttention inputs")

        seq_len = q.tt.shape[-2]
        dim = q.tt.shape[-1]
        if (seq_len % 32) != 0 or (dim % 32) != 0:
            raise ValueError("CausalAttention expects tile-aligned seq_len and dim")

        scale = self.scale if self.scale is not None else 1.0 / math.sqrt(dim)

        scores = q.matmul(k, transpose_b=True)
        if scale != 1.0:
            scale_tt = ttnn.full_like(scores.tt, fill_value=scale)
            scores = scores * Tensor(scale_tt, compute_kernel_config=q.compute_kernel_config)

        mask_tt = ttnn.full_like(scores.tt, fill_value=-1e9)
        mask_tt = ttnn.triu(mask_tt, diagonal=1)
        scores = scores + Tensor(mask_tt, compute_kernel_config=q.compute_kernel_config)

        attn = scores.softmax(dim=-1)
        return attn @ v


class TransformerBlock:
    def __init__(self, dim: int, *, heads: int = 1, ffn_mult: int = 4, dtype=None, device=None):
        self.dim = dim
        self.heads = heads
        self.ffn_mult = ffn_mult
        self.attn = MultiHeadAttention(dim, heads, dtype=dtype, device=device)
        self.q = self.attn.q
        self.k = self.attn.k
        self.v = self.attn.v
        self.o = self.attn.o
        self.ffn1 = Linear(dim, dim * ffn_mult, dtype=dtype, device=device)
        self.ffn2 = Linear(dim * ffn_mult, dim, dtype=dtype, device=device)

    def __call__(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x, dtype=self.q.dtype, device=self.q.weight.device)
        else:
            require_same_device(x, self.q.weight, name="TransformerBlock")

        attn_out = self.attn(x)

        y = self._postnorm(x + attn_out)
        ffn_out = self.ffn2(self.ffn1(y).relu())
        z = self._postnorm(y + ffn_out)
        return z

    def _postnorm(self, x: Tensor) -> Tensor:
        return x.tanh()

    def parameters(self):
        return [
            self.q.weight,
            self.k.weight,
            self.v.weight,
            self.o.weight,
            self.ffn1.weight,
            self.ffn2.weight,
        ]


class MultiHeadAttention:
    def __init__(self, dim: int, heads: int, *, dtype=None, device=None):
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        if (self.head_dim % 32) != 0:
            raise ValueError("head_dim must be a multiple of 32 for TILE_LAYOUT")

        self.q = Linear(dim, dim, dtype=dtype, device=device)
        self.k = Linear(dim, dim, dtype=dtype, device=device)
        self.v = Linear(dim, dim, dtype=dtype, device=device)
        self.o = Linear(dim, dim, dtype=dtype, device=device)
        self.attn = CausalAttention()

    def _split_heads(self, x: Tensor) -> Tensor:
        b, s, d = x.tt.shape
        x = x.permute((0, 2, 1))
        x = x.reshape((b, self.heads, self.head_dim, s))
        x = x.permute((0, 1, 3, 2))
        return x

    def _merge_heads(self, x: Tensor) -> Tensor:
        b, h, s, dh = x.tt.shape
        x = x.permute((0, 1, 3, 2))
        x = x.reshape((b, h * dh, s))
        x = x.permute((0, 2, 1))
        return x

    def __call__(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            x = Tensor(x, dtype=self.q.dtype, device=self.q.weight.device)
        else:
            require_same_device(x, self.q.weight, name="MultiHeadAttention")
        if len(x.tt.shape) != 3:
            raise ValueError("MultiHeadAttention expects 3D input (B, S, D)")
        if x.tt.shape[-1] != self.dim:
            raise ValueError("input dim mismatch in MultiHeadAttention")

        q = self._split_heads(self.q(x))
        k = self._split_heads(self.k(x))
        v = self._split_heads(self.v(x))
        attn_out = self.attn(q, k, v)
        merged = self._merge_heads(attn_out)
        return self.o(merged)

    def parameters(self):
        return [self.q.weight, self.k.weight, self.v.weight, self.o.weight]
