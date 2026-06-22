"""Export a seeded reference model to a flat binary the C++ loads, plus a test input
and the PyTorch reference logits for that input.

Outputs (into --outdir, default reference/artifacts/):
  weights.bin     header + all parameters as float32, in the canonical order below.
  input.bin       int32 T, then T int32 token ids.
  ref_logits.bin  int32 T, int32 vocab, then T*vocab float32 (PyTorch forward output).

The canonical weight order MUST match GPT::load in src/model.cpp exactly:
  wte, wpe,
  per layer: ln_1.w, ln_1.b, c_attn.w, c_attn.b, c_proj.w, c_proj.b,
             ln_2.w, ln_2.b, mlp.c_fc.w, mlp.c_fc.b, mlp.c_proj.w, mlp.c_proj.b,
  ln_f.w, ln_f.b, lm_head.w

File header: b"MGPT", int32 version=1, int32 {n_layer, n_head, n_embd, block_size, vocab_size}.
"""

import argparse
import os
import struct

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

VERSION = 1


def f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().contiguous().float().numpy().ravel()


def _ordered_params(model: GPT):
    """The model's parameters in the canonical order shared by the C++ loader, the
    weight file, and the gradient file. Yields the nn.Parameter objects themselves."""
    yield model.wte.weight             # (V, C)
    yield model.wpe.weight             # (block_size, C)
    for blk in model.h:
        yield blk.ln_1.weight
        yield blk.ln_1.bias
        yield blk.attn.c_attn.weight   # (3C, C)
        yield blk.attn.c_attn.bias     # (3C,)
        yield blk.attn.c_proj.weight   # (C, C)
        yield blk.attn.c_proj.bias     # (C,)
        yield blk.ln_2.weight
        yield blk.ln_2.bias
        yield blk.mlp.c_fc.weight      # (4C, C)
        yield blk.mlp.c_fc.bias        # (4C,)
        yield blk.mlp.c_proj.weight    # (C, 4C)
        yield blk.mlp.c_proj.bias      # (C,)
    yield model.ln_f.weight
    yield model.ln_f.bias
    yield model.lm_head.weight         # (V, C)


def canonical_tensors(model: GPT):
    """Parameter values in canonical order."""
    for p in _ordered_params(model):
        yield f32(p)


def canonical_grads(model: GPT):
    """Parameter gradients in canonical order (after a backward pass)."""
    for p in _ordered_params(model):
        assert p.grad is not None, "missing grad — did loss.backward() run?"
        yield f32(p.grad)


def write_weights(path, model: GPT):
    cfg = model.cfg
    with open(path, "wb") as f:
        f.write(b"MGPT")
        f.write(struct.pack("<i", VERSION))
        f.write(struct.pack("<5i", cfg.n_layer, cfg.n_head, cfg.n_embd,
                            cfg.block_size, cfg.vocab_size))
        for arr in canonical_tensors(model):
            f.write(arr.astype("<f4").tobytes())


def write_input(path, ids: np.ndarray):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(ids)))
        f.write(ids.astype("<i4").tobytes())


def write_logits(path, logits: np.ndarray):
    T, V = logits.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", T, V))
        f.write(logits.astype("<f4").tobytes())


def write_grads(path, loss: float, model: GPT):
    # float32 loss, then every parameter gradient concatenated in canonical order —
    # matching the C++ `grads` subcommand output.
    with open(path, "wb") as f:
        f.write(struct.pack("<f", loss))
        for arr in canonical_grads(model):
            f.write(arr.astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "artifacts"))
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--seqlen", type=int, default=12)
    # Config (small by default so the check runs fast).
    ap.add_argument("--n_layer", type=int, default=3)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--n_embd", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--vocab_size", type=int, default=65)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                    block_size=args.block_size, vocab_size=args.vocab_size)
    model = GPT(cfg).eval()

    assert args.seqlen <= cfg.block_size, "seqlen must be <= block_size"
    ids = np.random.randint(0, cfg.vocab_size, size=args.seqlen)
    targets = np.random.randint(0, cfg.vocab_size, size=args.seqlen)

    # Forward (grad-enabled), cross-entropy loss, backward — the reference gradients.
    model.zero_grad(set_to_none=False)
    logits_t = model(torch.tensor(ids, dtype=torch.long))           # (T, V)
    loss = F.cross_entropy(logits_t, torch.tensor(targets, dtype=torch.long))
    loss.backward()
    logits = logits_t.detach().numpy()

    write_weights(os.path.join(args.outdir, "weights.bin"), model)
    write_input(os.path.join(args.outdir, "input.bin"), ids)
    write_input(os.path.join(args.outdir, "targets.bin"), targets)
    write_logits(os.path.join(args.outdir, "ref_logits.bin"), logits)
    write_grads(os.path.join(args.outdir, "ref_grads.bin"), float(loss.item()), model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"exported model: {n_params} params, config={cfg}")
    print(f"  input seqlen={args.seqlen}, logits shape={logits.shape}")
    print(f"  reference loss={loss.item():.6f}")
    print(f"  artifacts in {args.outdir}")


if __name__ == "__main__":
    main()
