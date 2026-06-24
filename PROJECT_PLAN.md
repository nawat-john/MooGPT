# Tiny LLM from Scratch in C++ — Project Plan

> **Primary goal: learning.** This project exists to *understand* how a language
> model works down to the arithmetic. Every component is built by hand in C++ and
> verified against a PyTorch reference. We optimize for clarity and correctness
> first, speed last.

---

## 1. Goals & Non-Goals

**Goals**
- Build a small GPT-style (decoder-only) transformer **entirely in C++**, no ML framework.
- Implement the full math by hand: forward pass, backpropagation, and the optimizer.
- Train it to hold **basic English conversation** with a fixed persona: a *child-like
  character* — simple vocabulary, sweet/cute tone, never rude, limited knowledge.
- Understand every step well enough to explain *why* each gradient is what it is.

**Non-Goals (keep scope tiny on purpose)**
- Not chasing benchmark quality or ChatGPT-level ability.
- No mixed precision, no distributed training, no fancy kernels — at first.
- No external ML libs (PyTorch/TensorFlow/ggml) in the C++ code. PyTorch is used
  **only** as a *reference oracle* for verification, never linked into the build.

**Definition of done:** the model, trained on our data, produces coherent, on-tone,
clean replies to simple prompts, and we can explain/verify every computation in it.

---

## 2. Guiding Principles

1. **CPU-first, fp32-first.** Get it correct and understood before fast. GPU is a
   stretch phase, not a starting point.
2. **Verify every layer.** Nothing is "done" until its forward output and its
   gradients match the PyTorch reference within tolerance.
3. **One concept per milestone.** Don't implement attention and backprop and the
   optimizer in the same sitting. Land each, verify, then move on.
4. **Write notes as you go.** Keep `docs/notes.md` — the derivations matter more
   than the code for the learning goal.

---

## 3. Architecture Decisions

| Choice | Decision | Why |
|---|---|---|
| Model type | Decoder-only transformer (GPT-style) | Simplest path to a chat/generate loop |
| Precision | fp32 everywhere | Easiest to reason about & debug |
| Compute | CPU first (naive loops, then optional BLAS) | Understand the math before the hardware |
| Tokenizer | Char-level first → BPE later | Char-level removes a whole moving part early |
| Activation | GELU | Matches GPT-2 reference |
| Norm | LayerNorm (pre-norm) | Standard, stable, easy to differentiate |
| Position | Learned positional embeddings | Simpler than RoPE for v1 |

**Starter hyperparameters (Phase 4 char-level model, trains on CPU):**
`n_layer=4, n_embd=128, n_head=4, block_size(context)=128, vocab≈char set`
→ roughly 1–3M parameters. Scale up only after the pipeline works end to end.

---

## 4. Project Structure

```
tiny-llm/
├── CMakeLists.txt
├── README.md
├── src/
│   ├── tensor.h / tensor.cpp        # Tensor: shape, contiguous fp32 buffer, views
│   ├── ops.h / ops.cpp              # matmul, softmax, gelu, layernorm — fwd + bwd
│   ├── attention.h / attention.cpp  # causal multi-head self-attention
│   ├── mlp.h / mlp.cpp              # 2 linears + GELU
│   ├── block.h / block.cpp         # one transformer block (pre-norm + residual)
│   ├── model.h / model.cpp         # embeddings → blocks → final norm → lm_head
│   ├── loss.h / loss.cpp           # cross-entropy fwd + bwd
│   ├── optimizer.h / optimizer.cpp # AdamW
│   ├── tokenizer.h / tokenizer.cpp # char-level, later BPE
│   ├── dataloader.h / dataloader.cpp
│   └── main.cpp                    # subcommands: train | generate | check
├── reference/                      # PyTorch — verification ONLY, not built into C++
│   ├── model.py                    # same architecture, source of truth
│   ├── export_weights.py           # dump weights to a flat binary the C++ loads
│   └── check.py                    # compare C++ logits & grads vs autograd
├── data/
│   ├── prepare_corpus.py           # TinyStories / Shakespeare → train.txt / val.txt
│   ├── generate_dialogues.py       # synthetic persona dialogues via LLM API
│   └── build_vocab.py
├── tests/
│   └── test_ops.cpp                # unit tests + finite-difference grad checks
└── docs/
    └── notes.md                    # derivations, gotchas, learnings
```

---

## 5. Phased Roadmap

Each phase lists **what you build**, **what you learn**, and the **verification gate**
that must pass before moving on.

### Phase 0 — Foundations
- **Build:** CMake skeleton; a `Tensor` class (shape + row-major fp32 buffer);
  basic ops (`matmul`, elementwise add/mul) with unit tests.
- **Learn:** row-major memory layout, why indexing is `i*cols + j`, how matmul maps
  to nested loops, cache-friendliness basics.
- **Gate:** `test_ops` passes; a hand-checked 2×3 · 3×2 matmul is correct.

### Phase 1 — Tokenizer & Data Pipeline
- **Build:** char-level tokenizer (build vocab, `encode`/`decode`); dataloader that
  serves random `(x, y)` batches of length `block_size` with a train/val split.
- **Learn:** text → integer sequences, the context window, next-token targets being
  the input shifted by one.
- **Gate:** `decode(encode(text)) == text`; batches have correct shapes and the
  target is the input shifted by one position.

### Phase 2 — Forward Pass (inference)
- **Build:** token + positional embeddings → N×{LayerNorm → causal multi-head
  attention → LayerNorm → MLP, each with residual} → final LayerNorm → lm_head logits.
  Add a sampling loop (temperature, top-k) so you can *generate* even with random weights.
- **Learn:** the entire transformer forward math — especially scaled dot-product
  attention, the causal mask, and softmax.
- **Gate:** load identical weights exported from `reference/model.py`; C++ logits
  match PyTorch logits within ~1e-4.

### Phase 3 — Loss & Backward Pass *(the core learning phase)*
- **Build:** cross-entropy forward + backward; then backprop through **every** op —
  lm_head/linear, GELU, softmax, attention, LayerNorm, embeddings. Accumulate
  gradients into each parameter.
- **Learn:** backpropagation by hand, the chain rule through each module; the
  attention backward is the trickiest — budget extra time and write the derivation
  in `notes.md`.
- **Gate:** **two checks must both pass:**
  1. *Finite-difference check* inside C++ on a tiny tensor.
  2. *PyTorch autograd check* — same weights & input, every parameter gradient
     matches within tolerance (~1e-3 relative).

### Phase 4 — Optimizer & Training Loop
- **Build:** AdamW from scratch (first/second moments, bias correction, weight decay);
  training loop (forward → loss → backward → step → zero-grad); optional warmup +
  cosine LR schedule. Train the char-level model on TinyStories or Shakespeare.
- **Learn:** what optimization actually does; reading a loss curve; overfitting a
  single batch on purpose as a sanity test.
- **Gate:** loss on a single batch drives to ~0 (proves the loop is wired correctly);
  full-data training loss decreases steadily; generated text becomes word-like.

### Phase 5 — The Conversational Character
- **Build:** switch to / add a small BPE tokenizer for English; format data with role
  tokens (`<user>` / `<bot>`); assemble the dataset (see §6) and train; evaluate
  qualitatively for coherence, on-tone cuteness, and cleanliness.
- **Learn:** dialogue data formatting, how special tokens delimit turns, sampling
  strategy's effect on personality, simple decoding tricks.
- **Gate:** the model takes a `<user>` turn and produces a `<bot>` reply that is
  on-topic-enough, in the child-like sweet voice, and free of rude content.

### Phase 6 — Performance & GPU *(stretch / optional)*
- **Build:** profile hotspots; speed up matmul (OpenBLAS/Eigen on CPU, or hand-written
  CUDA + cuBLAS on the **local RTX 3050**); optionally add bf16. All training stays on
  the single 3050 — no cloud GPU. (3050 is Ampere, compute capability 8.6; modern CUDA
  works fully.)
- **Learn:** GPU programming, kernel basics, column-major vs row-major with cuBLAS,
  mixed-precision pitfalls, working within a small VRAM budget.
- **Gate:** GPU path produces the same loss/outputs as the verified CPU path.

---

## 6. Data Plan (English)

Three layers, each with a different job — quality and consistent voice beat volume
for a tiny model.

1. **Base language ability** — *TinyStories* (`roneneldan/TinyStories` on HuggingFace).
   Simple vocabulary at a young-child level; matches the persona naturally.
2. **Conversational format** — *DailyDialog* (`daily_dialog`). Clean everyday
   multi-turn dialogue; teaches turn-taking. Filter for length and tone.
3. **Persona voice (the key layer)** — **synthetic data generated by an LLM API.**
   No off-the-shelf set nails "child-like, cute, polite, limited knowledge," so we
   generate it. Lock a system prompt that fixes the tone, vocabulary level, and safety
   rules, and mass-produce short `<user>`/`<bot>` dialogues. Target ~10k–50k to start.

**Formatting** — one consistent scheme everywhere, e.g.:
```
<user> why is the sky blue?
<bot> hmm... i'm not totally sure, but i think it's just really pretty! like someone painted it for us.
```
**Safety pass** — run a final profanity/keyword filter over all data even after
controlled generation, so nothing slips through.

**Volume note (important on a single RTX 3050)** — VRAM limits *model size, batch
size, and context length*, **not** dataset size. The dataset lives on disk and streams
in batch by batch, so a big corpus never blows up VRAM; it only costs more wall-clock
time to train through. On one 3050 you are bounded by *time*, not data volume.
Practical sizing:
- **Don't download all of TinyStories** (it's ~millions of stories / GB-scale). Take a
  slice — tens of MB up to a few hundred MB of clean text is plenty for a char-level
  tiny model. Stream it from disk.
- **DailyDialog** is already small (~13k dialogues, a few MB) — use it filtered, as-is.
- **Synthetic persona** (~10k–50k short dialogues, a few MB) is the part worth your
  effort. Small, cheap, and it's what gives the model its voice.
- Keep the trained model small to match the hardware: char-level, ~1–10M params for
  Phase 4; for the Phase 5 chat model stay modest (~10–30M, context 128–256). Expect
  longer training runs rather than bigger data.

**How to get it**
1. `data/prepare_corpus.py` — download a TinyStories slice + Shakespeare from HuggingFace,
   clean, and write `train.txt` / `val.txt`.
2. `data/generate_dialogues.py` — call an LLM API with the locked persona system prompt
   to mass-produce the `<user>`/`<bot>` dialogues; adjustable count.
3. `data/build_vocab.py` — build the char vocab (Phase 4) or train a small BPE (Phase 5).

---

## 7. Verification Strategy (the learning backbone)

The PyTorch reference in `reference/` is the safety net that makes "from scratch"
feasible. It is **never** part of the C++ build — it only produces ground truth:

- `model.py` defines the *same* architecture in PyTorch.
- `export_weights.py` dumps a known set of weights to a flat binary the C++ loads,
  so both implementations run on identical parameters.
- `check.py` compares: (a) forward logits, (b) loss value, (c) every parameter
  gradient — against C++ output for the same input.

Add finite-difference gradient checks directly in `tests/` too, so correctness
doesn't depend on having PyTorch handy. **Rule: a layer isn't done until both its
forward and its backward match the reference.**

---

## 8. Milestone Checklist

- [x] P0: Tensor + matmul + tests green
- [x] P1: tokenizer round-trips; batches correct
- [x] P2: C++ forward logits == PyTorch (random weights loaded)
- [x] P3: all gradients pass finite-diff **and** autograd checks
- [x] P4: single-batch loss → ~0; real training loss decreasing; word-like samples
- [x] P5: persona model holds a basic, cute, clean conversation
- [ ] P6 (stretch): GPU path matches CPU path

---

## 9. Learning Resources

- **llm.c** (Karpathy) — GPT-2/3 training in pure C/CUDA. The closest reference to
  this project's end state; great for the CUDA phase.
- **llama2.c** (Karpathy) — single-file C inference; ideal model for clean,
  readable forward-pass code.
- **"Let's build GPT from scratch"** (Karpathy, video) — line-by-line transformer
  build; pair it with Phase 2–4.
- **TinyStories paper** (Eldan & Li) — why small + simple + clean data works for
  tiny models; the basis of our data strategy.
- **ggml** (Gerganov) — a C tensor library to study for memory/layout ideas later.

---

## 10. Suggested First Tasks for Claude Code

1. Scaffold the repo per §4 with a working CMake build and an empty test target.
2. Implement `Tensor` + `matmul` + the Phase 0 unit tests; make them pass.
3. Stand up `reference/model.py` and `export_weights.py` early — the verification
   harness should exist before Phase 2, not after.

> Keep each PR scoped to a single phase, and update `docs/notes.md` with the math
> derivation for whatever was implemented. The notes are part of the deliverable.
