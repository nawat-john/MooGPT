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

No build system exists yet — CMake is planned but not written. Once scaffolded, document the
actual `cmake`/`ctest` invocations and the `main` subcommands (`train`/`generate`/`check`) here.
Do not invent commands before the build files exist.

## Data plan (Phase 5)

Three layers (PROJECT_PLAN.md §6): TinyStories slice for base language, DailyDialog for
conversational turn-taking, and **LLM-generated synthetic persona dialogues** (the key layer that
gives the model its child-like voice). Stream large corpora from disk — dataset size does not
affect VRAM, only wall-clock time. Run a profanity/keyword safety filter over all data, including
the synthetic set.
