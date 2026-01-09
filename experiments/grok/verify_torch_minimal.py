"""
Minimal PyTorch transformer for step-by-step comparison with TTNN.
Matches verify_minimal.yaml config exactly.

Config: p=97, batch=16, dim=32, heads=1, ffn_mult=2, layers=1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math

# Match TTNN config exactly (must be multiples of 32)
P = 97
BATCH = 32
SEQ = 32  # pad_to
DIM = 64
HEADS = 2
FFN_MULT = 2
LAYERS = 1
SEED = 42
LR = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-6
WD = 0.01
ANSWER_POS = 3

# Vocab: p tokens + div_token + eq_token, padded to multiple of 32
BASE_VOCAB = P + 2  # 99
VOCAB = ((BASE_VOCAB + 31) // 32) * 32  # 128

torch.manual_seed(SEED)
np.random.seed(SEED)

def save_tensor(t, path):
    """Save tensor as raw float32 binary for easy comparison."""
    t_f32 = t.detach().cpu().float()
    t_np = t_f32.numpy()
    t_np.tofile(path)
    print(f"Saved {path}: shape={list(t.shape)}, norm={t_f32.norm().item():.6f}")

def load_tensor(path, shape):
    """Load tensor from raw BF16 binary."""
    data = np.fromfile(path, dtype=np.uint16)
    t = torch.from_numpy(data.view(np.float16).astype(np.float32)).view(shape)
    return t

class TransformerBlockLN(nn.Module):
    """Single transformer block with pre-LN."""
    def __init__(self, dim, heads, ffn_mult):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.ln1 = nn.LayerNorm(dim, eps=1e-5)
        self.wq = nn.Linear(dim, dim, bias=True)
        self.wk = nn.Linear(dim, dim, bias=True)
        self.wv = nn.Linear(dim, dim, bias=True)
        self.wo = nn.Linear(dim, dim, bias=True)

        self.ln2 = nn.LayerNorm(dim, eps=1e-5)
        self.ffn1 = nn.Linear(dim, dim * ffn_mult, bias=True)
        self.ffn2 = nn.Linear(dim * ffn_mult, dim, bias=True)

    def forward(self, x):
        B, S, D = x.shape

        # Pre-LN attention
        h = self.ln1(x)
        q = self.wq(h).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, S, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        out = self.wo(out)
        x = x + out

        # Pre-LN FFN with GELU
        h = self.ln2(x)
        ffn = self.ffn2(F.gelu(self.ffn1(h)))
        return x + ffn

class MinimalTransformer(nn.Module):
    """Minimal transformer matching TTNN PersistentGrokGPT2LN."""
    def __init__(self, vocab, seq, dim, heads, ffn_mult, n_layers):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(seq, dim)
        self.blocks = nn.ModuleList([
            TransformerBlockLN(dim, heads, ffn_mult) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(dim, eps=1e-5)
        self.out = nn.Linear(dim, vocab, bias=True)

    def forward(self, tokens):
        B, S = tokens.shape
        pos_idx = torch.arange(S, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.tok(tokens) + self.pos(pos_idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.out(x)

def mod_div(x, y, p):
    return (x * pow(y, p - 2, p)) % p

def create_batch(p, batch_size, seq_len, seed):
    """Create a deterministic batch matching TTNN."""
    rng = np.random.RandomState(seed)

    # Generate all possible (x, y) pairs
    all_pairs = [(x, y) for x in range(p) for y in range(1, p)]
    rng.shuffle(all_pairs)

    # Sample batch
    tokens = np.zeros((batch_size, seq_len), dtype=np.int64)
    targets = np.zeros(batch_size, dtype=np.int64)

    div_token = p
    eq_token = p + 1

    for i in range(batch_size):
        x, y = all_pairs[i % len(all_pairs)]
        z = mod_div(x, y, p)
        tokens[i, 0] = x
        tokens[i, 1] = div_token
        tokens[i, 2] = y
        tokens[i, 3] = eq_token
        targets[i] = z

    return torch.tensor(tokens), torch.tensor(targets)

def main():
    out_dir = "/tmp/verify_torch_minimal"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Config: p={P}, batch={BATCH}, seq={SEQ}, dim={DIM}, heads={HEADS}")
    print(f"        ffn_mult={FFN_MULT}, layers={LAYERS}, vocab={VOCAB}")
    print()

    # Create model with PyTorch-like init
    model = MinimalTransformer(VOCAB, SEQ, DIM, HEADS, FFN_MULT, LAYERS)

    # Initialize like TTNN (N(0,1) for embeddings, kaiming for linear)
    with torch.no_grad():
        nn.init.normal_(model.tok.weight, mean=0, std=1.0)
        nn.init.normal_(model.pos.weight, mean=0, std=1.0)
        # Linear layers use default kaiming init

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(BETA1, BETA2),
                            eps=EPS, weight_decay=WD)

    # Save initial weights
    save_tensor(model.tok.weight, f"{out_dir}/init_tok_weight.bin")
    save_tensor(model.pos.weight, f"{out_dir}/init_pos_weight.bin")
    save_tensor(model.out.weight, f"{out_dir}/init_out_weight.bin")
    save_tensor(model.out.bias, f"{out_dir}/init_out_bias.bin")

    # Create batch
    tokens, targets = create_batch(P, BATCH, SEQ, SEED)
    save_tensor(tokens.float(), f"{out_dir}/tokens.bin")
    save_tensor(targets.float(), f"{out_dir}/targets.bin")

    print(f"\nTokens shape: {tokens.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Sample tokens[0]: {tokens[0, :5].tolist()}")
    print(f"Sample target[0]: {targets[0].item()}")
    print()

    # Save initial weights for comparison
    tok_w0 = model.tok.weight.detach().clone()
    out_w0 = model.out.weight.detach().clone()

    # Training loop - just 5 steps for comparison
    for step in range(1, 6):
        opt.zero_grad()

        logits = model(tokens)
        loss = F.cross_entropy(logits[:, ANSWER_POS, :], targets)

        loss.backward()

        # Compute gradient norm
        grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
        w_norm = sum(p.norm().item()**2 for p in model.parameters())**0.5

        # Save weights BEFORE optimizer step
        tok_before = model.tok.weight.detach().clone()
        out_before = model.out.weight.detach().clone()

        print(f"Step {step}: loss={loss.item():.6f}, grad_norm={grad_norm:.6f}")
        print(f"  tok_grad_norm={model.tok.weight.grad.norm().item():.6f}")
        print(f"  out_grad_norm={model.out.weight.grad.norm().item():.6f}")

        opt.step()

        # Compute weight change
        tok_diff = (model.tok.weight - tok_before).norm().item()
        out_diff = (model.out.weight - out_before).norm().item()
        print(f"  tok_weight_change={tok_diff:.6f}")
        print(f"  out_weight_change={out_diff:.6f}")

        # Compute total change from init
        tok_total = (model.tok.weight - tok_w0).norm().item()
        out_total = (model.out.weight - out_w0).norm().item()
        print(f"  tok_total_from_init={tok_total:.6f}")
        print(f"  out_total_from_init={out_total:.6f}")

    print(f"\nOutputs saved to {out_dir}")
    print("Copy to howard and compare with TTNN outputs")

if __name__ == "__main__":
    main()
