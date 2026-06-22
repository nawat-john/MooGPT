"""Compare C++ logits against the PyTorch reference logits (the Phase 2 gate).

Both files use the format: int32 T, int32 vocab, then T*vocab float32 (row-major).

Usage:
    python check.py [--ref ref_logits.bin] [--cpp cpp_logits.bin] [--atol 1e-4]

Exit code 0 if max abs difference <= atol, else 1.
"""

import argparse
import os
import struct

import numpy as np

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def load_logits(path):
    with open(path, "rb") as f:
        T, V = struct.unpack("<ii", f.read(8))
        data = np.frombuffer(f.read(), dtype="<f4")
    if data.size != T * V:
        raise ValueError(f"{path}: expected {T*V} floats, got {data.size}")
    return data.reshape(T, V)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(DEFAULT_DIR, "ref_logits.bin"))
    ap.add_argument("--cpp", default=os.path.join(DEFAULT_DIR, "cpp_logits.bin"))
    ap.add_argument("--atol", type=float, default=1e-4)
    args = ap.parse_args()

    ref = load_logits(args.ref)
    cpp = load_logits(args.cpp)

    if ref.shape != cpp.shape:
        print(f"SHAPE MISMATCH: ref {ref.shape} vs cpp {cpp.shape}")
        raise SystemExit(1)

    diff = np.abs(ref - cpp)
    max_abs = float(diff.max())
    # Relative to the logit magnitude, for context.
    denom = np.maximum(np.abs(ref), 1e-6)
    max_rel = float((diff / denom).max())

    print(f"logits shape : {ref.shape}")
    print(f"max abs diff : {max_abs:.3e}")
    print(f"max rel diff : {max_rel:.3e}")
    print(f"tolerance    : {args.atol:.3e}")

    if max_abs <= args.atol:
        print("PASS: C++ logits match PyTorch within tolerance.")
        raise SystemExit(0)
    print("FAIL: C++ logits differ from PyTorch beyond tolerance.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
