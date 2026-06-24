# MooGPT — A Tiny LLM from Scratch in C++

A small GPT-style (decoder-only) transformer built **entirely by hand in C++** — the tensor
class, forward pass, backpropagation, AdamW optimizer, and tokenizers are all implemented from
scratch with **no ML framework linked into the build**. The end goal is a tiny model trained to
hold basic, sweet, child-like English conversation — a little character called **Moo**.

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
- **Compute:** CPU-first with naive loops, then OpenMP multi-core parallelism (~4.1× on 16 cores,
  Phase 6a), plus an optional, verified GPU/CUDA forward path on an RTX 3050 (Phase 6b).
- **Tokenizer:** char-level (Phases 0–4), then a from-scratch byte-level BPE with role tokens
  (`<user>`/`<bot>`/`<eot>`) for the Phase 5 chat model.

Starter hyperparameters (char-level model, trains on CPU):
`n_layer=4, n_embd=128, n_head=4, block_size=128` → roughly 1–3M parameters.

## Verification: the core discipline

The PyTorch code in `reference/` is a **verification oracle only** — it is never compiled into the
C++ build. The workflow:

1. `reference/model.py` defines the *same* architecture in PyTorch.
2. `reference/export_weights.py` dumps a fixed set of weights to a flat binary the C++ loads, so
   both run on identical parameters.
3. `reference/check.py` / `check_grads.py` compare C++ output vs PyTorch autograd: forward logits,
   loss, and **every parameter gradient**.

**A layer is not "done" until both its forward output and its gradients match the reference**
(~1e-4 for logits, ~1e-3 relative for gradients). C++ unit tests also include finite-difference
gradient checks so correctness doesn't depend on having PyTorch available.

## Project status

All phases are complete, including the Phase 6 stretch goals: CPU OpenMP parallelism (~4.1× on
16 cores) and a verified GPU/CUDA forward path on an RTX 3050. The model trains end to end and the
Phase 5 `chat` command holds a short, on-tone, clean conversation as Moo.

### Roadmap

| Phase | Goal | Gate |
|---|---|---|
| **P0** ✅ | Tensor + matmul + unit tests | Tests green; hand-checked matmul correct |
| **P1** ✅ | Char tokenizer + dataloader | `decode(encode(x))==x`; batches shaped right |
| **P2** ✅ | Forward pass (inference) | C++ logits match PyTorch within ~1e-4 |
| **P3** ✅ | Loss + backprop | All gradients pass finite-diff **and** autograd checks |
| **P4** ✅ | AdamW + training loop | Single-batch loss → ~0; real loss decreases; word-like samples |
| **P5** ✅ | Conversational persona | BPE + role tokens; holds a cute, clean `<user>`/`<bot>` chat |
| **P6a** ✅ | Performance: CPU OpenMP *(stretch)* | matmul/linear + attention (`H*T`-way) parallelized; ~4.1× on 16 cores, gates unchanged |
| **P6b** ✅ | GPU/CUDA forward *(stretch)* | hand-written kernels on RTX 3050 (CUDA 12.8); logits match CPU/PyTorch ~5e-7 (inference only) |

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for the full design, data plan, and learning resources,
and [`docs/notes.md`](docs/notes.md) for the per-phase math derivations.

## Structure

```
src/        C++ implementation: tensor, ops, attention, mlp, block, model, loss,
            optimizer, tokenizer (char), bpe (byte-level BPE), dataloader + main.cpp
            (subcommands: demo | check | grads | generate | train | chat)
reference/  PyTorch verification oracle — NOT built into the C++
data/       persona spec + corpus tools: PERSONA.md, persona_seed.txt,
            generate_dialogues.py, safety_filter.py, tiny.txt
tests/      C++ unit tests + finite-difference gradient checks (test_bpe = Phase 5)
docs/       notes.md — math derivations and learnings
```

## Building

A `CMakeLists.txt` is provided. With CMake installed:

```
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

If CMake isn't available, build directly with a C++17 compiler (e.g. g++). The full CLI:

```
g++ -std=c++17 -O2 -fopenmp -static -I src \
    src/tensor.cpp src/ops.cpp src/tokenizer.cpp src/bpe.cpp src/dataloader.cpp \
    src/attention.cpp src/mlp.cpp src/block.cpp src/model.cpp src/loss.cpp \
    src/optimizer.cpp src/main.cpp -o build/moogpt.exe
build/moogpt.exe demo     # hand-checked matmul demo
```

`-fopenmp` enables the Phase 6a multi-core speedup (matmul / linear over output rows, attention
over the flattened `(head, query-row)` space; ~4.1× on 16 cores); it's optional — without it the
build is the identical serial CPU path. `OMP_NUM_THREADS` controls the thread count at run time.

The optional Phase 6b GPU forward path is a single file, `src/forward_cuda.cu`, built with `nvcc`
and verified against the same PyTorch oracle (`reference/check.py`). See [`CLAUDE.md`](CLAUDE.md)
Phase 6b for the exact build (CUDA 12.8, `-arch=sm_86`) and toolchain notes.

Each test suite is a standalone target that exits 0 when all checks pass — see
[`CLAUDE.md`](CLAUDE.md) for the per-suite build/run commands and the PyTorch verification gates.

## Training & chatting with Moo (Phase 5)

```
# 1. Generate the persona corpus (deterministic, no API key needed) and safety-filter it
python data/generate_dialogues.py -n 8000 -o data/persona.txt
python data/safety_filter.py data/persona.txt          # -> data/persona.clean.txt

# 2. Train the BPE persona model (writes build/moo.bin + build/moo.bin.bpe)
build/moogpt.exe train data/persona.clean.txt --bpe --vocab 1024 \
    --steps 1500 --batch 16 --block 64 --lr 1e-3 --out build/moo.bin

# 3. Chat — single-turn or interactive
build/moogpt.exe chat build/moo.bin build/moo.bin.bpe "hi! how are you?"
build/moogpt.exe chat build/moo.bin build/moo.bin.bpe     # interactive (empty line to quit)
```

Sample replies from the trained model:

```
you> hello!
moo> hello hello! what a sweet day to say hi!
you> what is your name?
moo> my name is moo! i am a little friend. what is your name?
you> tell me a story
moo> okay! once a little sat on a cozy little nest and watched the stars. then it had
     a happy nap. the end! did you like it?
```

Voice, format, and safety are rock-solid; topic-matching is hit-or-miss (expected of a ~3M-param
model at 1500 steps) — a "train longer / bigger" lever, not a correctness gap.

Moo's whole personality lives in the data: [`data/PERSONA.md`](data/PERSONA.md) locks the character,
the exact `<user> …<eot><bot> …<eot>` format, and the generation prompt. `generate_dialogues.py`
mass-produces on-tone dialogues from curated templates; `safety_filter.py` is a final keyword pass.

## License

TBD.
