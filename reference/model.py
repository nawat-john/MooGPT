"""PyTorch reference model — the verification oracle for the C++ implementation.

This file is NEVER compiled into or linked against the C++ build. It defines the *same*
architecture as src/, so we can export identical weights and compare forward logits
(Phase 2) and, later, gradients (Phase 3).

Architecture (GPT-2 / nanoGPT style, pre-norm):
    tok_emb + pos_emb
      -> N x { x + attn(ln_1(x)); x + mlp(ln_2(x)) }
      -> ln_f
      -> lm_head (untied, no bias)

Deliberate choices that MUST match the C++ side exactly:
  * GELU uses the exact erf form (approximate='none') == std::erf in C++.
  * LayerNorm eps = 1e-5, biased variance (torch default).
  * Attention is written out explicitly (matmul + scale + causal mask + softmax)
    rather than F.scaled_dot_product_attention, to mirror attention.cpp.
  * lm_head is a separate weight, not tied to wte.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 64
    block_size: int = 16
    vocab_size: int = 65

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x):  # x: (T, C)
        T, C = x.shape
        H, d = self.cfg.n_head, self.cfg.head_dim
        qkv = self.c_attn(x)                      # (T, 3C)
        q, k, v = qkv.split(C, dim=-1)            # each (T, C)
        # (T, C) -> (H, T, d)
        q = q.view(T, H, d).transpose(0, 1)
        k = k.view(T, H, d).transpose(0, 1)
        v = v.view(T, H, d).transpose(0, 1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d)   # (H, T, T)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v                               # (H, T, d)
        y = y.transpose(0, 1).contiguous().view(T, C)    # (T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate="none")  # exact erf GELU
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx):  # idx: (T,) long
        T = idx.shape[0]
        pos = torch.arange(T, dtype=torch.long)
        x = self.wte(idx) + self.wpe(pos)  # (T, C)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)  # (T, vocab)
