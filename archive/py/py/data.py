import torch


def _next_multiple(value: int, multiple: int = 32) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _sieve_primes(max_n: int):
    if max_n < 2:
        return []
    sieve = [True] * (max_n + 1)
    sieve[0] = False
    sieve[1] = False
    p = 2
    while p * p <= max_n:
        if sieve[p]:
            step = p
            start = p * p
            sieve[start : max_n + 1 : step] = [False] * (((max_n - start) // step) + 1)
        p += 1
    return [i for i, is_prime in enumerate(sieve) if is_prime]


class PrimeSequenceDataset:
    def __init__(self, max_n: int, seq_len: int, *, seed: int = 0):
        if max_n < 2:
            raise ValueError("max_n must be >= 2")
        if (seq_len % 32) != 0:
            raise ValueError("seq_len must be a multiple of 32")
        self.max_n = int(max_n)
        self.seq_len = int(seq_len)
        self.vocab_size = _next_multiple(self.max_n + 1, 32)
        self.primes = _sieve_primes(self.max_n)
        if len(self.primes) < (self.seq_len + 1):
            raise ValueError("not enough primes for seq_len")
        self._generator = torch.Generator().manual_seed(seed)

    def sample_batch(self, batch_size: int):
        if (batch_size % 32) != 0:
            raise ValueError("batch_size must be a multiple of 32")
        max_start = len(self.primes) - (self.seq_len + 1)
        starts = torch.randint(0, max_start + 1, (batch_size,), generator=self._generator)
        sequences = [
            torch.tensor(self.primes[s : s + self.seq_len + 1], dtype=torch.int64) for s in starts.tolist()
        ]
        batch = torch.stack(sequences, dim=0)
        x = batch[:, :-1]
        y = batch[:, 1:]
        return x, y


class PrimeDataLoader:
    def __init__(self, dataset: PrimeSequenceDataset, *, batch_size: int, num_batches: int = None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_batches = None if num_batches is None else int(num_batches)

    def __iter__(self):
        count = 0
        while self.num_batches is None or count < self.num_batches:
            yield self.dataset.sample_batch(self.batch_size)
            count += 1


def _mod_div(x: int, y: int, p: int) -> int:
    return (x * pow(y, p - 2, p)) % p


class ModularDivisionDataset:
    def __init__(self, *, p: int = 97, train_frac: float = 0.5, pad_to: int = 32, seed: int = 0):
        if p <= 2:
            raise ValueError("p must be > 2")
        if not (0.0 < train_frac < 1.0):
            raise ValueError("train_frac must be in (0, 1)")
        if pad_to < 4 or (pad_to % 32) != 0:
            raise ValueError("pad_to must be >= 4 and a multiple of 32")
        self.p = int(p)
        self.div_token = self.p
        self.eq_token = self.p + 1
        self.base_vocab_size = self.p + 2
        self.vocab_size = _next_multiple(self.base_vocab_size, 32)
        self.seq_len = int(pad_to)
        self.answer_pos = 3

        xs = []
        ys = []
        zs = []
        for x in range(self.p):
            for y in range(1, self.p):
                xs.append(x)
                ys.append(y)
                zs.append(_mod_div(x, y, self.p))
        self._x_all = torch.tensor(xs, dtype=torch.int64)
        self._y_all = torch.tensor(ys, dtype=torch.int64)
        self._z_all = torch.tensor(zs, dtype=torch.int64)

        total = self._x_all.shape[0]
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(total, generator=gen)
        split = int(total * train_frac)
        self.train_idx = perm[:split]
        self.val_idx = perm[split:]

    def _sample_indices(self, batch_size: int, *, split: str, generator: torch.Generator):
        if (batch_size % 32) != 0:
            raise ValueError("batch_size must be a multiple of 32")
        idx_pool = self.train_idx if split == "train" else self.val_idx
        if idx_pool.numel() == 0:
            raise ValueError(f"{split} split is empty")
        rand = torch.randint(0, idx_pool.numel(), (batch_size,), generator=generator)
        return idx_pool[rand]

    def sample_batch(self, batch_size: int, *, split: str = "train", generator: torch.Generator = None):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        generator = generator or torch.Generator()
        idx = self._sample_indices(batch_size, split=split, generator=generator)
        x = self._x_all[idx]
        y = self._y_all[idx]
        z = self._z_all[idx]
        div_tok = torch.full_like(x, self.div_token)
        eq_tok = torch.full_like(x, self.eq_token)
        tokens = torch.zeros((batch_size, self.seq_len), dtype=torch.int64)
        tokens[:, 0] = x
        tokens[:, 1] = div_tok
        tokens[:, 2] = y
        tokens[:, 3] = eq_tok
        targets = z.unsqueeze(1)
        return tokens, targets

    def get_split_tensors(self, split: str):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        idx = self.train_idx if split == "train" else self.val_idx
        x = self._x_all[idx]
        y = self._y_all[idx]
        z = self._z_all[idx]
        div_tok = torch.full_like(x, self.div_token)
        eq_tok = torch.full_like(x, self.eq_token)
        tokens = torch.zeros((idx.numel(), self.seq_len), dtype=torch.int64)
        tokens[:, 0] = x
        tokens[:, 1] = div_tok
        tokens[:, 2] = y
        tokens[:, 3] = eq_tok
        targets = z.unsqueeze(1)
        return tokens, targets

    @property
    def train_size(self) -> int:
        return int(self.train_idx.numel())

    @property
    def val_size(self) -> int:
        return int(self.val_idx.numel())


class ModularDivisionDataLoader:
    def __init__(self, dataset: ModularDivisionDataset, *, batch_size: int, split: str = "train", num_batches=None):
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.split = split
        self.num_batches = None if num_batches is None else int(num_batches)
        self._generator = torch.Generator()

    def __iter__(self):
        count = 0
        while self.num_batches is None or count < self.num_batches:
            yield self.dataset.sample_batch(self.batch_size, split=self.split, generator=self._generator)
            count += 1
