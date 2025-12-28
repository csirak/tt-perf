import torch
import ttnn
from typing import Optional, Tuple


_TTNN_DTYPE = ttnn.bfloat16
_DEFAULT_COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi3,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
)


def _assert_tile_shape(shape: Tuple[int, ...]) -> None:
    if any((d % 32) != 0 for d in shape):
        raise ValueError(f"all dims must be multiples of 32 for TILE_LAYOUT, got {shape}")


def _torch_dtype_from_tt(tt_dtype) -> torch.dtype:
    if tt_dtype == ttnn.bfloat16:
        return torch.bfloat16
    if tt_dtype == ttnn.float32:
        return torch.float32
    return torch.float32


def resolve_tt_dtype(dtype):
    if dtype is None:
        return _TTNN_DTYPE
    if dtype in (ttnn.bfloat16, ttnn.float32):
        return dtype
    if dtype in (torch.bfloat16, torch.float32):
        return ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32
    if isinstance(dtype, str):
        key = dtype.lower()
        if key in ("bf16", "bfloat16"):
            return ttnn.bfloat16
        if key in ("fp32", "float32"):
            return ttnn.float32
    raise ValueError("dtype must be bf16/fp32, torch dtype, or ttnn dtype")


def resolve_torch_dtype(dtype) -> torch.dtype:
    if dtype is None:
        return get_torch_dtype()
    if dtype in (torch.bfloat16, torch.float32):
        return dtype
    return _torch_dtype_from_tt(resolve_tt_dtype(dtype))


def _to_tt(x, *, dtype=None, device=None):
    if isinstance(x, ttnn.Tensor):
        if device is not None and device != x.device():
            raise ValueError("device mismatch for ttnn.Tensor input")
        return x
    tt_dtype = resolve_tt_dtype(dtype)
    if device is None:
        raise RuntimeError("device must be provided for TTNN ops")
    tt_device = device
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=_torch_dtype_from_tt(tt_dtype))
    _assert_tile_shape(tuple(x.shape))
    return ttnn.from_torch(x, device=tt_device, layout=ttnn.TILE_LAYOUT, dtype=tt_dtype)


def _to_torch(x):
    return ttnn.to_torch(x)


def get_torch_dtype() -> torch.dtype:
    return _torch_dtype_from_tt(_TTNN_DTYPE)


class Tensor:
    def __init__(
        self,
        data: torch.Tensor,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
        requires_grad: bool = False,
        dtype=None,
        device=None,
        compute_kernel_config=_DEFAULT_COMPUTE_KERNEL_CONFIG,
    ):
        if isinstance(data, ttnn.Tensor):
            if device is not None and device != data.device():
                raise ValueError("device mismatch for ttnn.Tensor input")
            self.device = data.device()
        else:
            if device is None:
                raise RuntimeError("device must be provided for TTNN ops")
            self.device = device
        self.tt = _to_tt(data, dtype=dtype, device=self.device)
        self._torch_cache: Optional[torch.Tensor] = None
        self.grad: Optional[torch.Tensor] = None
        self.requires_grad = requires_grad
        self.compute_kernel_config = compute_kernel_config
        self._meta = {}
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self) -> str:
        shape = getattr(self.tt, "shape", None)
        return f"Tensor(shape={tuple(shape) if shape is not None else None}, requires_grad={self.requires_grad})"

    def to_torch(self) -> torch.Tensor:
        if self._torch_cache is None:
            self._torch_cache = _to_torch(self.tt)
        return self._torch_cache

    def set_compute_kernel_config(self, compute_kernel_config) -> None:
        self.compute_kernel_config = compute_kernel_config

    def estimate_flops(self) -> int:
        shape = tuple(getattr(self.tt, "shape", ()))
        if not shape:
            numel = 1
        else:
            numel = 1
            for d in shape:
                numel *= d

        if self._op == "matmul":
            meta = self._meta.get("matmul")
            if meta is not None:
                return meta.get("flops", 0)
            return 0

        if self._op in ("add", "mul", "tanh", "relu"):
            return numel

        if self._op == "softmax":
            return 5 * numel

        if self._op == "logsoftmax":
            return 6 * numel

        return 0

    def zero_grad(self) -> None:
        self.grad = None

    def _ensure_tt_grad(self, grad) -> ttnn.Tensor:
        if isinstance(grad, ttnn.Tensor):
            return grad
        return _to_tt(grad, device=self.device)

    def _add_grad(self, grad) -> None:
        if not self.requires_grad:
            return
        grad_tt = self._ensure_tt_grad(grad)
        if self.grad is None:
            self.grad = grad_tt
        else:
            self.grad = ttnn.add(self.grad, grad_tt)

    def grad_to_torch(self) -> Optional[torch.Tensor]:
        if self.grad is None:
            return None
        return _to_torch(self.grad)

    def backward(self, grad: Optional[torch.Tensor] = None) -> None:
        if grad is None:
            shape = tuple(self.tt.shape)
            numel = 1
            for d in shape:
                numel *= d
            if numel != 1:
                raise ValueError("grad must be specified for non-scalar outputs")
            grad = ttnn.ones_like(self.tt)
        else:
            grad = self._ensure_tt_grad(grad)

        topo = []
        visited = set()

        def build(v: "Tensor"):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = grad
        for v in reversed(topo):
            v._backward()

    def __add__(self, other: "Tensor") -> "Tensor":
        other = _as_tensor(other, device=self.device, compute_kernel_config=self.compute_kernel_config)
        out_tt = ttnn.add(self.tt, other.tt)
        out = Tensor(
            out_tt,
            (self, other),
            "add",
            self.requires_grad or other.requires_grad,
            compute_kernel_config=_merge_compute_kernel_config(self, other),
        )

        def _backward():
            if self.requires_grad:
                self._add_grad(out.grad)
            if other.requires_grad:
                other._add_grad(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other: "Tensor") -> "Tensor":
        other = _as_tensor(other, device=self.device, compute_kernel_config=self.compute_kernel_config)
        out_tt = ttnn.mul(self.tt, other.tt)
        out = Tensor(
            out_tt,
            (self, other),
            "mul",
            self.requires_grad or other.requires_grad,
            compute_kernel_config=_merge_compute_kernel_config(self, other),
        )

        def _backward():
            if self.requires_grad:
                self._add_grad(ttnn.mul(other.tt, out.grad))
            if other.requires_grad:
                other._add_grad(ttnn.mul(self.tt, out.grad))

        out._backward = _backward
        return out

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def matmul(self, other: "Tensor", *, transpose_a: bool = False, transpose_b: bool = False) -> "Tensor":
        other = _as_tensor(other, device=self.device, compute_kernel_config=self.compute_kernel_config)
        compute_kernel_config = _merge_compute_kernel_config(self, other)
        if compute_kernel_config is None:
            out_tt = ttnn.matmul(self.tt, other.tt, transpose_a=transpose_a, transpose_b=transpose_b)
        else:
            out_tt = ttnn.matmul(
                self.tt,
                other.tt,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
                compute_kernel_config=compute_kernel_config,
            )
        out = Tensor(
            out_tt,
            (self, other),
            "matmul",
            self.requires_grad or other.requires_grad,
            compute_kernel_config=compute_kernel_config,
        )
        try:
            a_shape = list(self.tt.shape)
            b_shape = list(other.tt.shape)
            if transpose_a and len(a_shape) >= 2:
                a_shape[-1], a_shape[-2] = a_shape[-2], a_shape[-1]
            if transpose_b and len(b_shape) >= 2:
                b_shape[-1], b_shape[-2] = b_shape[-2], b_shape[-1]
            out_shape = list(out_tt.shape)
            if len(out_shape) >= 2:
                m = out_shape[-2]
                n = out_shape[-1]
                k = a_shape[-1]
                batch = 1
                for d in out_shape[:-2]:
                    batch *= d
                flops = 2 * batch * m * n * k
            else:
                flops = 0
            out._meta["matmul"] = {
                "flops": flops,
                "transpose_a": transpose_a,
                "transpose_b": transpose_b,
            }
        except Exception:
            pass

        def _transpose_last_two(tt):
            dims = list(range(len(tt.shape)))
            dims[-1], dims[-2] = dims[-2], dims[-1]
            return ttnn.permute(tt, dims)

        def _reduce_to_shape(grad, target_shape):
            grad_shape = list(grad.shape)
            target_shape = list(target_shape)
            while len(grad_shape) > len(target_shape):
                grad = ttnn.sum(grad, dim=0, keepdim=False)
                grad_shape = list(grad.shape)
            for i in reversed(range(len(target_shape))):
                if target_shape[i] == 1 and grad_shape[i] != 1:
                    grad = ttnn.sum(grad, dim=i, keepdim=True)
                    grad_shape[i] = 1
            return grad

        def _backward():
            if self.requires_grad:
                grad_self = ttnn.matmul(out.grad, other.tt, transpose_b=not transpose_b)
                if transpose_a:
                    grad_self = _transpose_last_two(grad_self)
                grad_self = _reduce_to_shape(grad_self, self.tt.shape)
                self._add_grad(grad_self)
            if other.requires_grad:
                grad_other = ttnn.matmul(self.tt, out.grad, transpose_a=not transpose_a)
                if transpose_b:
                    grad_other = _transpose_last_two(grad_other)
                grad_other = _reduce_to_shape(grad_other, other.tt.shape)
                other._add_grad(grad_other)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        out_tt = ttnn.tanh(self.tt)
        out = Tensor(out_tt, (self,), "tanh", self.requires_grad, compute_kernel_config=self.compute_kernel_config)

        def _backward():
            if self.requires_grad:
                y2 = ttnn.mul(out.tt, out.tt)
                one = ttnn.ones_like(out.tt)
                dy = ttnn.sub(one, y2)
                self._add_grad(ttnn.mul(dy, out.grad))

        out._backward = _backward
        return out

    def reshape(self, shape) -> "Tensor":
        shape_tuple = tuple(shape)
        out_tt = ttnn.reshape(self.tt, shape_tuple)
        out = Tensor(out_tt, (self,), "reshape", self.requires_grad, compute_kernel_config=self.compute_kernel_config)

        def _backward():
            if self.requires_grad:
                self._add_grad(ttnn.reshape(out.grad, tuple(self.tt.shape)))

        out._backward = _backward
        return out

    def permute(self, order) -> "Tensor":
        order_tuple = tuple(order)
        out_tt = ttnn.permute(self.tt, order_tuple)
        out = Tensor(out_tt, (self,), "permute", self.requires_grad, compute_kernel_config=self.compute_kernel_config)

        def _backward():
            if self.requires_grad:
                inverse = [i for i, _ in sorted(enumerate(order_tuple), key=lambda x: x[1])]
                self._add_grad(ttnn.permute(out.grad, inverse))

        out._backward = _backward
        return out

    def softmax(self, dim: int = -1) -> "Tensor":
        if self.compute_kernel_config is None:
            out_tt = ttnn.softmax(self.tt, dim=dim)
        else:
            out_tt = ttnn.softmax(self.tt, dim=dim, compute_kernel_config=self.compute_kernel_config)
        out = Tensor(out_tt, (self,), "softmax", self.requires_grad, compute_kernel_config=self.compute_kernel_config)

        def _backward():
            if self.requires_grad:
                dot = ttnn.sum(ttnn.mul(out.grad, out.tt), dim=dim, keepdim=True)
                grad_input = ttnn.mul(out.tt, ttnn.sub(out.grad, dot))
                self._add_grad(grad_input)

        out._backward = _backward
        return out

    def logsoftmax(self, dim: int = -1) -> "Tensor":
        if self.compute_kernel_config is None:
            softmax_tt = ttnn.softmax(self.tt, dim=dim)
        else:
            softmax_tt = ttnn.softmax(self.tt, dim=dim, compute_kernel_config=self.compute_kernel_config)
        out_tt = ttnn.log(softmax_tt)
        out = Tensor(out_tt, (self,), "logsoftmax", self.requires_grad, compute_kernel_config=self.compute_kernel_config)

        def _backward():
            if self.requires_grad:
                sum_grad = ttnn.sum(out.grad, dim=dim, keepdim=True)
                softmax_tt_local = ttnn.exp(out.tt)
                grad_input = ttnn.sub(out.grad, ttnn.mul(softmax_tt_local, sum_grad))
                self._add_grad(grad_input)

        out._backward = _backward
        return out

    def relu(self) -> "Tensor":
        out_tt = ttnn.relu(self.tt)
        out = Tensor(out_tt, (self,), "relu", self.requires_grad, compute_kernel_config=self.compute_kernel_config)

        def _backward():
            if self.requires_grad:
                mask = ttnn.gtz(self.tt)
                self._add_grad(ttnn.mul(out.grad, mask))

        out._backward = _backward
        return out


def require_same_device(*tensors, name: str):
    if not tensors:
        return
    for t in tensors:
        if not isinstance(t, Tensor):
            raise ValueError(f"{name} expects Tensor inputs")
    device = tensors[0].device
    for t in tensors[1:]:
        if t.device != device:
            raise ValueError(f"device mismatch in {name} inputs")


def _merge_compute_kernel_config(*tensors):
    configs = [t.compute_kernel_config for t in tensors if isinstance(t, Tensor) and t.compute_kernel_config is not None]
    if not configs:
        return None
    first = configs[0]
    if any(cfg != first for cfg in configs[1:]):
        raise ValueError("compute_kernel_config mismatch between tensors")
    return first


def tensor(
    data,
    requires_grad: bool = False,
    dtype=None,
    device=None,
    compute_kernel_config=_DEFAULT_COMPUTE_KERNEL_CONFIG,
) -> Tensor:
    return Tensor(
        data,
        requires_grad=requires_grad,
        dtype=dtype,
        device=device,
        compute_kernel_config=compute_kernel_config,
    )


def _as_tensor(x, *, device=None, dtype=None, compute_kernel_config=_DEFAULT_COMPUTE_KERNEL_CONFIG) -> Tensor:
    if isinstance(x, Tensor):
        if device is not None and x.device != device:
            raise ValueError("device mismatch between tensors")
        if compute_kernel_config is not None and x.compute_kernel_config is not None:
            if compute_kernel_config != x.compute_kernel_config:
                raise ValueError("compute_kernel_config mismatch between tensors")
        return x
    return Tensor(x, device=device, dtype=dtype, compute_kernel_config=compute_kernel_config)
