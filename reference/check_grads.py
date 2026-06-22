"""Compare C++ gradients against PyTorch autograd (the Phase 3 gate, part 2).

Reads the model config from weights.bin's header to reconstruct the canonical parameter
layout, then compares the per-parameter gradient blocks in ref_grads.bin (PyTorch
autograd) vs cpp_grads.bin (hand-written C++ backward). Both grad files are:
    float32 loss, then all parameter gradients (float32) concatenated in canonical order.

Usage:
    python check_grads.py [--weights ...] [--ref ...] [--cpp ...] [--rtol 1e-3] [--atol 1e-5]

Exit 0 only if the losses match and every parameter's gradient is allclose.
"""

import argparse
import os
import struct

import numpy as np

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def read_config(weights_path):
    with open(weights_path, "rb") as f:
        magic = f.read(4)
        if magic != b"MGPT":
            raise ValueError(f"bad magic in {weights_path}: {magic!r}")
        version = struct.unpack("<i", f.read(4))[0]
        n_layer, n_head, n_embd, block_size, vocab_size = struct.unpack("<5i", f.read(20))
    return dict(version=version, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                block_size=block_size, vocab_size=vocab_size)


def param_specs(cfg):
    """(name, numel) for every parameter, in canonical order."""
    C, V, B, L = cfg["n_embd"], cfg["vocab_size"], cfg["block_size"], cfg["n_layer"]
    specs = [("wte", V * C), ("wpe", B * C)]
    for l in range(L):
        p = f"h{l}."
        specs += [
            (p + "ln_1.w", C), (p + "ln_1.b", C),
            (p + "attn.c_attn.w", 3 * C * C), (p + "attn.c_attn.b", 3 * C),
            (p + "attn.c_proj.w", C * C), (p + "attn.c_proj.b", C),
            (p + "ln_2.w", C), (p + "ln_2.b", C),
            (p + "mlp.c_fc.w", 4 * C * C), (p + "mlp.c_fc.b", 4 * C),
            (p + "mlp.c_proj.w", C * 4 * C), (p + "mlp.c_proj.b", C),
        ]
    specs += [("ln_f.w", C), ("ln_f.b", C), ("lm_head.w", V * C)]
    return specs


def read_grads(path):
    with open(path, "rb") as f:
        loss = struct.unpack("<f", f.read(4))[0]
        data = np.frombuffer(f.read(), dtype="<f4")
    return loss, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(DEFAULT_DIR, "weights.bin"))
    ap.add_argument("--ref", default=os.path.join(DEFAULT_DIR, "ref_grads.bin"))
    ap.add_argument("--cpp", default=os.path.join(DEFAULT_DIR, "cpp_grads.bin"))
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-5)
    args = ap.parse_args()

    cfg = read_config(args.weights)
    specs = param_specs(cfg)
    total = sum(n for _, n in specs)

    ref_loss, ref = read_grads(args.ref)
    cpp_loss, cpp = read_grads(args.cpp)

    if ref.size != total or cpp.size != total:
        print(f"SIZE MISMATCH: expected {total} grads, ref={ref.size}, cpp={cpp.size}")
        raise SystemExit(1)

    print(f"config: {cfg}")
    print(f"loss  : ref={ref_loss:.6f}  cpp={cpp_loss:.6f}  |diff|={abs(ref_loss-cpp_loss):.3e}")
    print(f"{'parameter':<22} {'max_abs':>10} {'max_rel':>10}  status")
    print("-" * 56)

    ok = abs(ref_loss - cpp_loss) <= 1e-4
    if not ok:
        print("LOSS MISMATCH (> 1e-4)")

    worst_rel = 0.0
    off = 0
    for name, n in specs:
        a = ref[off:off + n]
        b = cpp[off:off + n]
        off += n
        diff = np.abs(a - b)
        max_abs = float(diff.max()) if n else 0.0
        denom = np.abs(a) + args.atol
        max_rel = float((diff / denom).max()) if n else 0.0
        worst_rel = max(worst_rel, max_rel)
        passed = np.allclose(b, a, rtol=args.rtol, atol=args.atol)
        ok = ok and passed
        print(f"{name:<22} {max_abs:>10.2e} {max_rel:>10.2e}  {'ok' if passed else 'FAIL'}")

    print("-" * 56)
    print(f"worst relative diff across all params: {worst_rel:.3e}")
    if ok:
        print("PASS: C++ gradients match PyTorch autograd.")
        raise SystemExit(0)
    print("FAIL: C++ gradients differ from PyTorch autograd.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
