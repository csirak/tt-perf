import ttnn

from grad.engine import Tensor


class SGD:
    def __init__(self, params, *, lr: float):
        self.params = list(params)
        if lr <= 0:
            raise ValueError("lr must be > 0")
        self.lr = float(lr)

    def step(self):
        for p in self.params:
            if not isinstance(p, Tensor):
                raise ValueError("SGD expects Tensor parameters")
            if p.grad is None:
                continue
            lr_tt = ttnn.full_like(p.grad, fill_value=self.lr)
            update = ttnn.mul(p.grad, lr_tt)
            p.tt = ttnn.sub(p.tt, update)
            p._torch_cache = None

    def zero_grad(self):
        for p in self.params:
            if isinstance(p, Tensor):
                p.zero_grad()


class Adam:
    def __init__(self, params, *, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = list(params)
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError("betas must be a tuple of (beta1, beta2)")
        beta1, beta2 = float(betas[0]), float(betas[1])
        if not (0.0 < beta1 < 1.0 and 0.0 < beta2 < 1.0):
            raise ValueError("betas must be in (0, 1)")
        self.lr = float(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = float(eps)
        self._state = {}
        self._step = 0

    def step(self):
        self._step += 1
        bias1 = 1.0 - (self.beta1 ** self._step)
        bias2 = 1.0 - (self.beta2 ** self._step)
        for p in self.params:
            if not isinstance(p, Tensor):
                raise ValueError("Adam expects Tensor parameters")
            if p.grad is None:
                continue
            state = self._state.get(id(p))
            if state is None:
                m = ttnn.full_like(p.grad, fill_value=0.0)
                v = ttnn.full_like(p.grad, fill_value=0.0)
                state = {
                    "m": m,
                    "v": v,
                    "lr": ttnn.full_like(p.grad, fill_value=self.lr),
                    "beta1": ttnn.full_like(p.grad, fill_value=self.beta1),
                    "beta2": ttnn.full_like(p.grad, fill_value=self.beta2),
                    "one_minus_beta1": ttnn.full_like(p.grad, fill_value=1.0 - self.beta1),
                    "one_minus_beta2": ttnn.full_like(p.grad, fill_value=1.0 - self.beta2),
                    "eps": ttnn.full_like(p.grad, fill_value=self.eps),
                }
                self._state[id(p)] = state

            m = state["m"]
            v = state["v"]
            m = ttnn.add(ttnn.mul(state["beta1"], m), ttnn.mul(state["one_minus_beta1"], p.grad))
            grad_sq = ttnn.mul(p.grad, p.grad)
            v = ttnn.add(ttnn.mul(state["beta2"], v), ttnn.mul(state["one_minus_beta2"], grad_sq))
            state["m"] = m
            state["v"] = v

            m_hat = ttnn.div(m, ttnn.full_like(p.grad, fill_value=bias1))
            v_hat = ttnn.div(v, ttnn.full_like(p.grad, fill_value=bias2))
            denom = ttnn.add(ttnn.sqrt(v_hat), state["eps"])
            step = ttnn.mul(state["lr"], ttnn.div(m_hat, denom))
            p.tt = ttnn.sub(p.tt, step)
            p._torch_cache = None

    def zero_grad(self):
        for p in self.params:
            if isinstance(p, Tensor):
                p.zero_grad()
