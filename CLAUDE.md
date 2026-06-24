# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository currently contains **only `PROJECT_PLAN.md`** — no source, build files, or
tests exist yet. The plan is the source of truth; read it in full before scaffolding anything.
The directory layout, hyperparameters, and roadmap below are *targets to build toward*, not
existing code. When something here disagrees with the actual files on disk, the files win —
update this document.

## What this project is

A small GPT-style (decoder-only) transformer built **entirely by hand in C++** — tensor class,
forward pass, backprop, and AdamW all from scratch, with **no ML framework linked into the build**.
The end goal is a tiny model trained to hold basic, sweet, child-like English conversation.

**The primary goal is learning, not benchmark quality.** This reshapes normal engineering
priorities — optimize in this order: *correctness → clarity/understanding → speed (last)*.
Concretely:
- Prefer naive, readable loops over clever/vectorized code until a later, explicit perf phase.
- fp32 everywhere; CPU-first. GPU/BLAS/bf16 are a stretch phase (Phase 6), not a starting point.
- One concept per change. Don't land attention, backprop, and the optimizer together — land one,
  verify it, then move on. Keep each change scoped to a single phase.
- Math derivations are a deliverable. Update `docs/notes.md` with the derivation for whatever you
  implement (especially gradients) — the notes matter as much as the code here.

## The verification workflow (the core discipline)

This is the most important thing to understand. PyTorch lives in `reference/` purely as a
**verification oracle** — it is *never* compiled or linked into the C++ build. The loop is:

1. `reference/model.py` defines the *same* architecture in PyTorch (source of truth).
2. `reference/export_weights.py` dumps a fixed set of weights to a flat binary that the C++ loads,
   so both implementations run on identical parameters.
3. `reference/check.py` compares C++ output vs PyTorch autograd: forward logits, loss, and **every
   parameter gradient**.

**Rule: a layer is not "done" until both its forward output and its gradients match the reference
within tolerance** (~1e-4 for forward logits, ~1e-3 relative for gradients). In addition, put
finite-difference gradient checks directly in C++ `tests/` so correctness doesn't depend on having
PyTorch available. Build the verification harness *before* Phase 2, not after.

## Target architecture

Decoder-only transformer: token + learned positional embeddings → N × {pre-LayerNorm → causal
multi-head self-attention → pre-LayerNorm → MLP (2 linears + GELU), each with a residual} → final
LayerNorm → lm_head logits. Tokenizer is char-level first, BPE later (Phase 5).

Starter hyperparameters (Phase 4 char-level model, trains on CPU):
`n_layer=4, n_embd=128, n_head=4, block_size=128, vocab≈char set` → ~1–3M params.

Planned source layout (see PROJECT_PLAN.md §4 for the full tree): `src/` holds one module per
concept (`tensor`, `ops`, `attention`, `mlp`, `block`, `model`, `loss`, `optimizer`, `tokenizer`,
`dataloader`) with `main.cpp` exposing `train | generate | check` subcommands. `reference/` is
PyTorch-only, `data/` has Python prep scripts, `tests/` has the C++ unit/grad-check target.

## Roadmap & gates

Work proceeds in phases (PROJECT_PLAN.md §5); each has a verification gate that must pass first:

- **P0** Tensor + matmul + unit tests green.
- **P1** Tokenizer round-trips (`decode(encode(x))==x`); batches shaped right, target = input shifted by one.
- **P2** C++ forward logits match PyTorch within ~1e-4 (random weights loaded from the reference).
- **P3** All gradients pass *both* finite-diff and autograd checks. Attention backward is the hardest — budget extra time.
- **P4** AdamW + training loop; single-batch loss drives to ~0 (proves wiring), real loss decreases, samples become word-like.
- **P5** BPE tokenizer + role tokens (`<user>`/`<bot>`); persona model holds a cute, clean conversation.
- **P6** (stretch) GPU/CUDA path on the local RTX 3050 produces the same loss/outputs as the verified CPU path.

## Build & run

**CMake is not installed on this machine** (checked Phase 0). `CMakeLists.txt` exists as the
intended build, but builds are currently done directly with **g++ 14.2.0** at
`C:\msys64\ucrt64\bin\g++.exe`. MSVC BuildTools 18 (`cl.exe`) is also present. To compile, prepend
the compiler to PATH: `$env:Path = "C:\msys64\ucrt64\bin;$env:Path"`.

**Two PATH gotchas, both learned in Phase 2 — important:**
1. **Build with `-static`.** Otherwise the exe depends on UCRT64 runtime DLLs and fails with
   `0xC0000135` (DLL not found) unless `C:\msys64\ucrt64\bin` is on PATH at *run* time.
2. **Do NOT have `ucrt64\bin` on PATH when running `python`.** msys ships its own `python.exe`
   (no numpy/torch) that shadows the real interpreter at
   `C:\Users\nawatpim\AppData\Local\Programs\Python\Python312\python.exe`. So: set the msys PATH
   only for the g++ compile step; run the static exe and python from a clean shell.

Build + run the test suites (each exits 0 = all checks passed):
```
g++ -std=c++17 -fopenmp -static -Wall -Wextra -Wpedantic -I src src/tensor.cpp src/ops.cpp tests/test_ops.cpp -o build/test_ops.exe
build/test_ops.exe        # Phase 0: tensor + ops

g++ -std=c++17 -fopenmp -static -Wall -Wextra -Wpedantic -I src src/tokenizer.cpp src/dataloader.cpp tests/test_data.cpp -o build/test_data.exe
build/test_data.exe       # Phase 1: tokenizer + dataloader

g++ -std=c++17 -fopenmp -static -Wall -Wextra -Wpedantic -I src src/tensor.cpp src/ops.cpp src/loss.cpp tests/test_grad.cpp -o build/test_grad.exe
build/test_grad.exe       # Phase 3: finite-difference gradient checks

g++ -std=c++17 -fopenmp -static -Wall -Wextra -Wpedantic -I src src/tensor.cpp src/optimizer.cpp tests/test_optim.cpp -o build/test_optim.exe
build/test_optim.exe      # Phase 4: AdamW (convex minimization + weight-decay behavior)

g++ -std=c++17 -fopenmp -static -Wall -Wextra -Wpedantic -I src src/bpe.cpp tests/test_bpe.cpp -o build/test_bpe.exe
build/test_bpe.exe        # Phase 5: BPE round-trip, special tokens, determinism, save/load
```
Build the full CLI (`demo`/`check`/`grads`/`generate`/`train`/`chat` all work):
```
g++ -std=c++17 -O2 -fopenmp -static -I src src/tensor.cpp src/ops.cpp src/tokenizer.cpp src/bpe.cpp src/dataloader.cpp \
    src/attention.cpp src/mlp.cpp src/block.cpp src/model.cpp src/loss.cpp src/optimizer.cpp src/main.cpp -o build/moogpt.exe
```
When new `.cpp` files land, add them to both the g++ command and the `moocore` library in
`CMakeLists.txt`. Once CMake is installed, prefer `cmake -S . -B build && cmake --build build`
then `ctest --test-dir build`.

**`-fopenmp` is the Phase 6 speedup** (matmul / linear / attention parallelized across cores;
~3.7× on 16 cores here). It is optional: drop the flag and the `#pragma omp` lines are ignored,
giving the identical serial CPU path. Control threads at run time with `OMP_NUM_THREADS`. The
parallel decomposition is race-free *and* keeps every reduction's order, so results match the
serial/PyTorch gates within the same tolerances (Phase 2 ~6e-7 logits, Phase 3 ~4e-4 grads) — see
`docs/notes.md` Phase 6. libgomp links statically under `-static`, so the exe stays self-contained.

The C++ test harness is dependency-free (no gtest/Catch): each `tests/*.cpp` counts checks and
returns a non-zero exit code on failure, so it gates cleanly. No single-test filter yet — add one
(e.g. an argv name match) if a suite grows large.

### Phase 2 verification gate (C++ forward vs PyTorch)

PyTorch 2.12.1+cpu is installed (`pip install torch --index-url .../cpu`). Run the gate from a
**clean shell** (no msys PATH tweak), after building `moogpt.exe`:
```
python reference/export_weights.py        # writes reference/artifacts/{weights,input,ref_logits}.bin
build/moogpt.exe check reference/artifacts/weights.bin reference/artifacts/input.bin reference/artifacts/cpp_logits.bin
python reference/check.py                  # PASS if max abs diff <= 1e-4 (currently ~6e-7)
```
`export_weights.py` takes `--n_layer/--n_head/--n_embd/--block_size/--vocab_size/--seqlen` to test
other configs. The weight binary format (`MGPT` magic + int32 version + 5 int32 config + float32
params in canonical order) is a contract defined identically in `export_weights.py::canonical_tensors`
and `model.cpp::GPT::load` — changing the parameter set means editing **both** in lockstep.

**Matching subtleties** (where two "correct" transformers diverge): exact erf GELU on both sides
(not tanh), LayerNorm biased variance + eps 1e-5, `nn.Linear` weight layout `(out,in)`, explicit
(non-fused) attention, fused-QKV split order `[Q|K|V]`, untied lm_head. See `docs/notes.md` Phase 2.

### Phase 3 verification gate (C++ gradients vs PyTorch autograd)

Two checks, both must pass. (1) C++ finite differences: `build/test_grad.exe`. (2) Autograd, from
a clean shell after building `moogpt.exe`:
```
python reference/export_weights.py     # now also writes targets.bin + ref_grads.bin (loss+grads)
build/moogpt.exe grads reference/artifacts/weights.bin reference/artifacts/input.bin reference/artifacts/targets.bin reference/artifacts/cpp_grads.bin
python reference/check_grads.py         # per-parameter compare, rtol 1e-3 (currently worst ~4e-4)
```
Backward design: every `*_backward` helper returns `dx` and **accumulates** (`+=`) into the param
grad buffers, so `model.zero_grad()` must run before `model.backward()`. `forward()` is non-const
(it caches activations); `parameters()`/`gradients()` expose the canonical-order tensors that both
the grad file format and the FD check rely on. Derivations for every op (incl. the attention
backward) are in `docs/notes.md` Phase 3.

### Phase 4 verification gate (training, no PyTorch needed)

Two parts. (1) The single-batch overfit sanity check — proves the forward→loss→backward→step→
zero_grad loop is wired:
```
build/moogpt.exe train data/tiny.txt --overfit --steps 500 --batch 4 --block 32 --lr 1e-3
# loss must collapse toward ~0 (observed 3.36 -> ~0.03); --overfit forces weight_decay=0
```
(2) Real training — full-corpus loss decreases steadily and decoded samples become word-like:
```
build/moogpt.exe train data/tiny.txt --steps 3000 --batch 16 --block 64 --lr 1e-3
```
Plus `build/test_optim.exe` (AdamW convex-minimization + decay behavior, dependency-free).
`train` flags: `--steps/--batch/--block/--layer/--head/--embd/--lr/--wd/--seed/--out/--overfit`;
the trained model is saved in the same MGPT format `load`/`generate` read. `data/tiny.txt` is a
small public-domain child-voice corpus. AdamW only weight-decays 2-D+ params (matrices/embeddings),
not biases/LayerNorm. Batch=1 model: train loops the B rows, scaling each row's `dlogits` by 1/B so
accumulation yields the mean-loss gradient. Derivations in `docs/notes.md` Phase 4.

## Data plan (Phase 5)

Three layers (PROJECT_PLAN.md §6): TinyStories slice for base language, DailyDialog for
conversational turn-taking, and **LLM-generated synthetic persona dialogues** (the key layer that
gives the model its child-like voice). Stream large corpora from disk — dataset size does not
affect VRAM, only wall-clock time. Run a profanity/keyword safety filter over all data, including
the synthetic set.
