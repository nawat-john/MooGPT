# MooGPT — A Tiny LLM from Scratch in C++

A small GPT-style (decoder-only) transformer built **entirely by hand in C++** — the tensor
class, forward pass, backpropagation, and AdamW optimizer are all implemented from scratch with
**no ML framework linked into the build**. The end goal is a tiny model trained to hold basic,
sweet, child-like English conversation.

> **The primary goal is learning, not benchmark quality.** This project exists to *understand* how
> a language model works down to the arithmetic. We optimize for correctness and clarity first,
> speed last.

## Why

Most "from scratch" LLM projects still lean on PyTorch or a tensor library. Here, every
computation — including the gradient of every layer — is written and verified by hand. PyTorch is
used **only** as a reference oracle to check our math, never as a dependency of the C++ code.

## How it works

A decoder-only transformer:

```
token + learned positional embeddings
  → N × { pre-LayerNorm → causal multi-head self-attention → residual
          pre-LayerNorm → MLP (Linear → GELU → Linear)       → residual }
  → final LayerNorm
  → lm_head → logits
```

- **Precision:** fp32 everywhere (easiest to reason about).
- **Compute:** CPU-first with naive loops; GPU/BLAS is a later, optional phase.
- **Tokenizer:** char-level first, BPE later.

Starter hyperparameters (char-level model, trains on CPU):
`n_layer=4, n_embd=128, n_head=4, block_size=128` → roughly 1–3M parameters.

## Verification: the core discipline

The PyTorch code in `reference/` is a **verification oracle only** — it is never compiled into the
C++ build. The workflow:

1. `reference/model.py` defines the *same* architecture in PyTorch.
2. `reference/export_weights.py` dumps a fixed set of weights to a flat binary the C++ loads, so
   both run on identical parameters.
3. `reference/check.py` compares C++ output vs PyTorch autograd: forward logits, loss, and **every
   parameter gradient**.

**A layer is not "done" until both its forward output and its gradients match the reference**
(~1e-4 for logits, ~1e-3 relative for gradients). C++ unit tests also include finite-difference
gradient checks so correctness doesn't depend on having PyTorch available.

## Project status

Early/planning stage. The repository currently contains the design document
([`PROJECT_PLAN.md`](PROJECT_PLAN.md)) and is being scaffolded per the roadmap below.

### Roadmap

| Phase | Goal | Gate |
|---|---|---|
| **P0** ✅ | Tensor + matmul + unit tests | Tests green; hand-checked matmul correct |
| **P1** ✅ | Char tokenizer + dataloader | `decode(encode(x))==x`; batches shaped right |
| **P2** | Forward pass (inference) | C++ logits match PyTorch within ~1e-4 |
| **P3** | Loss + backprop | All gradients pass finite-diff **and** autograd checks |
| **P4** | AdamW + training loop | Single-batch loss → ~0; real loss decreases |
| **P5** | Conversational persona | Holds a cute, clean `<user>`/`<bot>` conversation |
| **P6** | Performance & GPU *(stretch)* | GPU path matches the verified CPU path |

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full design, data plan, and learning resources.

## Planned structure

```
src/        C++ implementation (tensor, ops, attention, mlp, block, model,
            loss, optimizer, tokenizer, dataloader) + main.cpp
reference/  PyTorch verification oracle — NOT built into the C++
data/       Python corpus/dialogue/vocab prep scripts
tests/      C++ unit tests + finite-difference gradient checks
docs/       notes.md — math derivations and learnings
```

## Building

A `CMakeLists.txt` is provided. With CMake installed:

```
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

If CMake isn't available, build directly with a C++17 compiler (e.g. g++):

```
g++ -std=c++17 -Wall -Wextra -Wpedantic -I src \
    src/tensor.cpp src/ops.cpp tests/test_ops.cpp -o build/test_ops.exe
build/test_ops.exe        # exit 0 = all checks passed

g++ -std=c++17 -I src src/tensor.cpp src/ops.cpp src/main.cpp -o build/moogpt.exe
build/moogpt.exe demo     # hand-checked matmul demo
```

## License

TBD.
