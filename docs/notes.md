# Notes — derivations, gotchas, learnings

Running log kept per the plan. The math derivations matter as much as the code.

---

## Phase 0 — Foundations

### Row-major memory layout

A tensor's logical N-D shape is stored in one flat, contiguous `std::vector<float>`.
"Row-major" (C-style) means the **last axis is fastest-varying**: walking memory
linearly, the last index increments first.

For shape `(d0, d1, ..., d_{n-1})` the **stride** of axis `k` is the product of all
dimension sizes *after* it:

```
stride[k] = d_{k+1} * d_{k+2} * ... * d_{n-1}     (stride of the last axis = 1)
```

and the flat offset of a full index `(i0, i1, ..., i_{n-1})` is:

```
offset = sum_k i_k * stride[k]
```

Examples (verified in `tests/test_ops.cpp`):
- shape `(2,3)` → strides `(3,1)` → element `(i,j)` at `i*3 + j`. This is the famous
  `i*cols + j`.
- shape `(2,3,4)` → strides `(12,4,1)` → element `(1,2,3)` at `1*12 + 2*4 + 3 = 23`.

Why one flat buffer instead of nested `vector<vector<...>>`: contiguous memory is
cache-friendly, has no per-row pointer chasing, and is the layout BLAS/cuBLAS expect —
so the same buffer ports to fast kernels later without a reshape.

### matmul as nested loops

For `A (M,K) @ B (K,N) = C (M,N)`:

```
C[i,j] = sum_k A[i,k] * B[k,j]
```

The textbook order is `i, j, k` (compute each output element fully). We instead use
**`i, k, j`** with `C` zero-initialized and accumulate `C[i,j] += A[i,k] * B[k,j]`.
Mathematically identical, but the memory access pattern is better:

- In the inner `j` loop, `A[i,k]` is a fixed scalar (`aik`).
- `B[k,j]` walks `B`'s row `k` contiguously (`bp + k*N`, stride 1).
- `C[i,j]` walks `C`'s row `i` contiguously (`op + i*N`, stride 1).

So both the read of B and the write of C stride sequentially through memory → fewer
cache misses than `i,j,k`, where `B[k,j]` would jump by `N` each step (column walk).

### Phase 0 gate (hand-checked)

`A (2x3) @ B (3x2)`:

```
A = [[1,2,3],        B = [[ 7, 8],
     [4,5,6]]             [ 9,10],
                          [11,12]]

C[0,0] = 1*7 + 2*9  + 3*11 = 58
C[0,1] = 1*8 + 2*10 + 3*12 = 64
C[1,0] = 4*7 + 5*9  + 6*11 = 139
C[1,1] = 4*8 + 5*10 + 6*12 = 154
```

Matches the C++ output. **Gate passed.**

### Gotchas / decisions

- No external test framework — a tiny in-file harness (count checks, non-zero exit on
  failure) keeps the build dependency-free and failures obvious.
- `add`/`mul` require identical shapes (no broadcasting yet); add broadcasting only when
  a real op needs it, to avoid silent shape bugs early.
- Bounds/shape mismatches `abort()` with a message rather than throwing — loud and
  immediate is the right failure mode while learning.

---

## Phase 1 — Tokenizer & Data Pipeline

### Char-level tokenizer

The vocabulary is just the set of **distinct characters** in the corpus, sorted by byte
value. The id of a character is its index in that sorted list. So `encode` is a per-char
map lookup and `decode` is a per-id array lookup — `O(len)` both ways.

- Sorting makes ids **deterministic** across runs (independent of the order chars first
  appear), which matters because the model's embedding rows are keyed by id — a reshuffle
  would silently invalidate a saved checkpoint.
- Round-trip gate: `decode(encode(text)) == text`. Holds by construction for any text
  whose characters are all in the vocab; `encode` aborts loudly on an unknown char rather
  than dropping or guessing.
- Char-level deliberately removes a whole moving part (subword merging) until Phase 5.

### Next-token targets: y is x shifted by one

Language modeling trains the model to predict the next token at every position. So from
a window of `block_size + 1` tokens starting at corpus index `s`:

```
x[t] = data[s + t]        for t in [0, T)     # what the model sees
y[t] = data[s + t + 1]    for t in [0, T)     # what it must predict (= x shifted left 1)
```

One window yields both x and y; that's why a valid start must satisfy
`s + block_size <= len(data) - 1`, i.e. `s ∈ [0, len - block_size - 1]`.

Test trick: use the corpus `0,1,2,...,N` so each token's **value equals its position**.
Then the shift is checkable from values alone — `y[b,t] == x[b,t] + 1` — and contiguity
within a row is `x[b,t+1] == x[b,t] + 1`. No need to track which start index was drawn.

### Batching & splits

- A `Batch` is row-major `(B, T)` stored in two flat `vector<int>` (`x`, `y`), mirroring
  the `Tensor` layout philosophy. Token ids are integers, not fp32 — they index embedding
  rows later, so keeping them `int` avoids float-rounding id bugs.
- Train/val split is a **contiguous tail cut** (last `val_fraction` of tokens), not
  interleaved, so validation text is genuinely held out.
- The RNG (`std::mt19937`) is passed *into* `get_batch`, so determinism is the caller's
  choice — tests seed it to get reproducible batches; training can seed per run.

### Gotchas / decisions

- `std::uniform_int_distribution<size_t>(0, len - T - 1)` — the `-1` is essential, else
  the last target `data[s+T]` reads out of bounds.
- Windows are drawn **with replacement** (independent uniform starts), so a batch can in
  principle repeat a window. Fine for SGD; revisit if we want epoch-style shuffling.

---

## Phase 2 — Forward Pass (inference)

The whole transformer forward, fp32, single sequence `(T, C)`, batch=1. Matched against a
PyTorch oracle to **max abs diff ~6e-7** (gate is 1e-4).

### Architecture (must match `reference/model.py` exactly)

```
x = wte[tokens] + wpe[0..T-1]                      # (T, C)
repeat N times:
    x = x + attn(layernorm(x, ln_1))              # pre-norm + residual
    x = x + mlp (layernorm(x, ln_2))
x = layernorm(x, ln_f)
logits = x @ lm_head^T                             # (T, vocab), no bias
```

### Things that must match bit-for-bit-ish across C++/PyTorch

These are the subtle knobs where two "correct" transformers diverge past 1e-4:

- **GELU variant.** Exact erf form `0.5 x (1 + erf(x/√2))` on both sides (`std::erf` in C++,
  `F.gelu(..., approximate='none')` in torch). The tanh approximation would *not* match.
- **LayerNorm:** biased variance (÷C, not ÷(C−1)), eps = 1e-5 — torch's defaults.
- **Linear layout:** weight stored `(out, in)` like `nn.Linear`; `y = x @ Wᵀ + b`. Exporting
  torch weights as-is then drops straight into the C++ `linear`.
- **Attention written out explicitly** (matmul → scale by 1/√d → causal mask → softmax →
  weighted sum of V) rather than a fused kernel, so it mirrors `attention.cpp` op-for-op.
- **Fused QKV split order:** `c_attn` outputs `[Q | K | V]` along features; head h owns columns
  `[h·d, (h+1)·d)` within each block. Same on both sides (nanoGPT convention).
- **lm_head untied** from `wte` (separate weight) — simpler/explicit while bringing it up.

### Scaled dot-product attention + causal mask

`scores[i,j] = (q_i · k_j)/√d`; mask `j > i` to `-inf` so position i can only attend to
≤ i (autoregressive); softmax each row; `out_i = Σ_j p[i,j] v_j`. The `-inf` entries become
exactly 0 after softmax, so the C++ skips them (`j <= i`) — same result, less work.

### Verification harness (the oracle)

`reference/` is PyTorch, never linked into C++. Flow:

1. `export_weights.py` — seed a model, write `weights.bin` (header + params in canonical
   order), `input.bin` (token ids), and `ref_logits.bin` (torch forward output).
2. `moogpt check weights.bin input.bin cpp_logits.bin` — C++ loads weights, runs forward,
   dumps its logits.
3. `check.py` — compares `ref_logits.bin` vs `cpp_logits.bin`, max abs diff vs tolerance.

The binary format is the contract: `b"MGPT"`, int32 version, int32×5 config
`(n_layer, n_head, n_embd, block_size, vocab_size)`, then float32 params in the **canonical
order** defined identically in `export_weights.py::canonical_tensors` and
`model.cpp::GPT::load`. The C++ loader checks magic/version and asserts no trailing bytes —
a config/order mismatch surfaces immediately instead of as silent numeric garbage.

### Sampling (generate)

Take the last position's logits, divide by temperature, optional top-k (keep k largest via
`nth_element`, mask the rest to −inf), softmax, sample with `std::discrete_distribution`.
Context is cropped to the last `block_size` tokens each step.

### Gotchas / decisions

- **Static linking required to run the exe.** A plain g++ build depends on UCRT64 runtime
  DLLs (`libstdc++-6`, `libgcc_s_seh-1`, `libwinpthread-1`); running it without
  `C:\msys64\ucrt64\bin` on PATH fails with `0xC0000135` (DLL not found). Build with
  `-static` so the binary is self-contained.
- **Don't prepend `ucrt64\bin` to PATH when invoking `python`** — msys ships its own
  `python.exe` (no numpy/torch) and it shadows the real interpreter. Build C++ statically,
  then run python from a clean PATH. (This is why the gate is run in a shell *without* the
  msys PATH tweak.)
- Tolerance reality: max **abs** diff ~1e-6; max **rel** diff can be ~1e-3–1e-2 where a logit
  is near zero — abs diff is the meaningful gate for logits.

---

## Phase 3 — Loss & Backward Pass (the core phase)

Hand-derived backprop through every op. Verified two ways: C++ finite differences
(`test_grad`) and PyTorch autograd (`check_grads.py`). All 40 parameter tensors match
autograd to **worst relative diff ~4e-4**; loss matches to ~5e-7.

### Convention

Each backward helper takes upstream `dy` (shape of the op's output), returns `dx`, and
**accumulates** (`+=`) into the parameter-gradient buffers. Accumulation (not assignment)
is what lets a parameter reached by several paths sum its contributions — e.g. an
embedding row hit by a repeated token, or the residual stream. Callers `zero_grad()` once
before a backward pass. Modules cache the forward inputs their backward needs.

### Cross-entropy

Per position t with logits z and target y: `loss_t = logsumexp(z) − z[y]`, averaged over T.
The gradient is the clean softmax−onehot result:
```
dlogits[t,:] = (softmax(z_t) − onehot(y_t)) / T
```
The `/T` comes from the mean reduction and must be there for the magnitude to match torch.

### Linear  (y = x Wᵀ + b)

```
dx = dy W          dW += dyᵀ x          db += colsum(dy)
```
Implemented in one fused triple loop: for each (t,o), `dx[t,:] += dy·W[o,:]` and
`dW[o,:] += dy·x[t,:]` share the inner stride over `in`.

### GELU (exact erf)

```
d/dx [0.5 x (1+erf(x/√2))] = 0.5(1+erf(x/√2)) + x·(1/√(2π))·exp(−x²/2)
```
First term is the gelu "gate" (its own CDF), second is `x` times the standard normal PDF.

### LayerNorm

With `xhat = (x−μ)·rstd`, `rstd = 1/√(var+eps)`, `y = γ·xhat + β`:
```
dγ += Σ_t dy·xhat        dβ += Σ_t dy
dxhat = dy·γ
dx = rstd · (dxhat − mean(dxhat) − xhat·mean(dxhat·xhat))    # per row
```
The two subtracted means are the corrections for μ and var both depending on every x in
the row — the part people get wrong. mean/var are recomputed from x in backward (cheap,
avoids caching).

### Softmax (row-wise Jacobian-vector product)

```
dx[i] = y[i] · (dy[i] − Σ_j dy[j] y[j])
```
The shared `Σ_j dy[j] y[j]` is the only cross-term; everything else is elementwise.

### Attention backward (the hard one)

Forward per head: `S = (QKᵀ)·scale → causal mask → P = softmax(S) → O = P V`. Going
backward from `dO` (a slice of the c_proj input gradient):
```
dP[i,j] = Σ_e dO[i,e] V[j,e]          dV[j] += Σ_i P[i,j] dO[i]
dS = softmax_backward(P, dP)          # per row; masked entries have P=0 ⇒ contribute 0
dQ[i] += scale · Σ_j dS[i,j] K[j]     dK[j] += scale · Σ_i dS[i,j] Q[i]
```
Key points:
- **Causality is automatic in backward.** Masked positions (`j>i`) have `P=0`, so both
  `dV` and the softmax JVP yield 0 there — no separate masking needed in the backward.
- **The `scale` reappears** when differentiating `S = (QKᵀ)·scale`.
- `dQ/dK/dV` are scattered back into the fused `dQKV (T,3C)` at the head's columns, then
  one `linear_backward` through `c_attn` produces `dx` and the c_attn param grads.

### Block & model wiring

Pre-norm residual `x_out = x + sub(ln(x))` means the input gradient splits in two: the
direct residual path **plus** the path through the sublayer and its LayerNorm. So each
block does `dx = dx_residual + ln_backward(sub_backward(dx_residual))`, twice (mlp then
attn). At the top, the embedding gradient routes the per-position `dx[t]` into **both**
`d_wte[token[t]]` and `d_wpe[t]` (accumulated).

### Verification harness (Phase 3 additions)

- `export_weights.py` now also picks random `targets`, runs `F.cross_entropy` +
  `loss.backward()`, and writes `targets.bin` and `ref_grads.bin` (float32 loss + all
  param grads in canonical order).
- `moogpt grads weights input targets out` runs the C++ forward+loss+backward and writes
  the same format.
- `check_grads.py` reconstructs the parameter layout from the weight header and compares
  every parameter block (rtol 1e-3), reporting per-parameter max abs/rel diff.

### Gotchas / decisions

- **forward() is now non-const** (it caches activations for backward). `generate()` became
  non-const as a result. Inference pays the cache cost; fine for a tiny model.
- FD in fp32 is noisy — `test_grad` uses eps=1e-2 and loose tolerances (rtol 3e-2). It
  catches sign/factor/structure bugs; the autograd check is the tight one. The two are
  complementary, satisfying the plan's "both checks must pass" gate.
- Loss accumulated in `double` inside cross-entropy to keep the scalar stable across T.

---

## Phase 4 — Optimizer & Training Loop

AdamW from scratch + the training loop that drives forward → loss → backward → step →
zero_grad. Verified two ways: a self-contained `test_optim` (convex minimization +
weight-decay behavior, no PyTorch) and the plan's two-part training gate.

### AdamW

Adam keeps two running averages per parameter — the first moment `m` (mean gradient) and
the second moment `v` (mean squared gradient):
```
m = β1·m + (1−β1)·g
v = β2·v + (1−β2)·g²
```
Both start at 0, so early on they're biased toward 0. Bias correction divides them back
out (each divisor → 1 as t grows):
```
m̂ = m / (1 − β1^t)     v̂ = v / (1 − β2^t)
θ ← θ − lr · m̂ / (√v̂ + eps)
```
`m̂/√v̂` is a per-parameter adaptive step: directions with consistently large/noisy
gradients get smaller effective steps. Defaults β1=0.9, β2=0.999, eps=1e-8.

**The "W": decoupled weight decay.** Plain Adam adds L2 (`+wd·θ`) into `g`, which then runs
through the adaptive `/√v̂` denominator — so it is *not* uniform decay. AdamW instead
shrinks the parameter directly, independent of the moments:
```
θ ← θ − lr·wd·θ      (applied before the Adam step)
```
Per the GPT-2 convention we **only decay 2-D+ tensors** (weight matrices, embeddings), not
biases or LayerNorm gains/shifts — the optimizer keys this off `param.ndim() >= 2`.
`test_optim` checks all three behaviors: convex problems converge to the true optimum, a
matrix with zero grad decays as `(1−lr·wd)^t`, and a 1-D param with zero grad is untouched.

### Batching with a batch=1 model

`forward()`/`backward()` run one sequence at a time. To train on a `(B, T)` batch we loop
the B rows: forward, cross-entropy (which already averages over T), then **scale that row's
`dlogits` by 1/B** before `backward()`. Because backward *accumulates* into the grad
buffers, summing B rows each scaled by 1/B yields exactly the gradient of the mean loss
over the batch. `zero_grad()` runs once before the row loop, `opt.step()` once after.

### Initialization (training from scratch)

Until now parameters only ever came from `GPT::load`. `GPT::random_init` adds the GPT-2
recipe: `normal(0, 0.02)` for embeddings and linear weights, zeros for biases, ones for
LayerNorm gains. The two residual projections (`attn.c_proj`, `mlp.c_proj`) instead use
`0.02/√(2·n_layer)` — each block writes two residual contributions, so over n_layer blocks
the stream accumulates `2·n_layer` of them; this scaling keeps its variance ~constant with
depth (otherwise deep models blow up at init).

### LR schedule

Linear warmup (0 → lr over a few % of steps) then half-cosine decay to `0.1·lr`. Warmup
avoids the large, badly-conditioned early steps when Adam's `v` estimate is still noisy;
cosine decay anneals to fine-tune near the end. The overfit sanity run uses a constant lr.

### Verification gate (two parts)

1. **Overfit a single fixed batch** — `train --overfit` reuses one batch every step. Loss
   must collapse toward ~0; this proves the loop is wired (any break — a sign error, a
   missing `zero_grad`, a transposed grad — leaves loss stuck). Observed: 3.36 → ~0.03 in
   500 steps on `data/tiny.txt`. Weight decay is forced to 0 here (it would fight reaching 0).
2. **Real training** — full-corpus loss decreases steadily and decoded samples become
   word-like (spaces in sensible places, real short words, line breaks). `data/tiny.txt`
   is a small public-domain child-voice corpus (nursery rhymes / simple sentences) that
   suits both the char-level model and the eventual persona.

### Gotchas / decisions

- **Captured pointer stability.** `AdamW` stores the `parameters()` pointers and `step()`
  takes the `gradients()` pointers; both are addresses of GPT *member* tensors, which stay
  fixed even though `zero_grad()` reallocates the underlying grad buffers each step. So
  capturing them once is safe.
- **Checkpoint = the MGPT format.** `GPT::save` writes the exact header+canonical-order
  binary that `load` reads, so a trained model round-trips back in for `generate`. Loss is
  accumulated in `double` across the batch for a stable scalar.
- Speed is still last priority: one sequence at a time, naive loops. Fine for the tiny
  char model; Phase 6 is where this gets optimized.

---

## Phase 5 — The Conversational Character

Adds a from-scratch byte-level BPE tokenizer with role tokens, a conversation data format,
and a chat loop, then trains the child-like persona model.

### Byte-level BPE (`bpe.{h,cpp}`)

BPE = Byte-Pair Encoding. Start from the 256 raw byte values and repeatedly **merge the
most frequent adjacent token pair** into a new token, growing the vocab to a target size.
Encoding then replays those merges (lowest-rank/earliest-learned first) on each piece of
input; decoding concatenates each token's stored bytes.

- **Byte-level ⇒ always round-trips.** The base alphabet is all 256 bytes, so any input —
  unseen characters, raw binary, UTF-8 multibyte — encodes and decodes exactly. No `<unk>`.
  `test_bpe` fuzzes 500 random byte strings through `decode(encode(x)) == x`.
- **Pre-tokenization into byte-class "words."** Before training, text is split into maximal
  runs of one class (letters / digits / whitespace / other) and merges never cross a word
  boundary. This keeps merges meaningful (no token spanning a space into the next word) and
  makes training fast: we count pairs over the *unique* words weighted by frequency, and
  apply each merge only within words — orders of magnitude less work than scanning the whole
  corpus per merge.
- **Determinism.** Pair counts live in an ordered `std::map`; ties go to the smallest
  `(a,b)`. Same corpus ⇒ identical merges (checked in `test_bpe`).
- **Special / role tokens.** `<user>`, `<bot>`, `<eot>` are reserved strings given ids
  *after* the byte+merge vocab. `encode` splits them out first (longest-match), emits their
  id atomically, and BPE-encodes only the text between — so a role token is always exactly
  one token and can reliably delimit turns. They're also excluded from merge training.
- **Vocab id layout:** `[0,256)` bytes, `[256, 256+#merges)` merges, then the specials.
  `vocab_size()` includes the specials; the model's `cfg.vocab_size` is set from it.
- Save/load is a small text format (`MOOBPE 1`, the ordered merge pairs, the special
  strings); the vocab is rebuilt from the merges on load.

### Conversation format & chat loop

Canonical turn format (locked in `data/PERSONA.md`):
```
<user> {message}<eot><bot> {reply}<eot>
```
Training is plain next-token LM over the concatenated token stream — the DataLoader and GPT
from Phases 1–4 are unchanged; only the tokenizer differs (`train --bpe`). The model learns
to emit `<bot> …<eot>` after a `<user> …<eot>`. `chat` prompts with everything up to and
including the `<bot>` token, then samples until `<eot>` (the new `stop_id` in `generate`),
and decodes the span in between as the reply.

### Persona data (the voice)

The personality is entirely in the data. `data/PERSONA.md` locks the character ("Moo": sweet,
child-like, simple words, honest about not knowing, never unsafe), the exact format, and the
generation system prompt. `data/persona_seed.txt` is a hand-written seed (~45 dialogues) that
defines the format and wires/tests the pipeline.

The bulk corpus is produced by `data/generate_dialogues.py` → `data/persona.txt`. PROJECT_PLAN
§6 envisioned generating this layer with an LLM API; the script is instead a **dependency-free,
deterministic template generator** — it composes dialogues from curated per-topic templates with
slot fillers (animals, colors, foods, numbers, weather, the honest "hmm... i'm not sure"
pattern for unknowns, …), mixing single- and multi-turn (1–4 turns). This keeps Moo's voice
locked, needs no API key or network, and reproduces exactly from `--seed`. Trade-off: less
lexical variety than a real LLM would give, but for a tiny model trained to *adopt a consistent
voice* that's a feature — and the API path stays documented in PERSONA.md for a richer run.
Default run is ~8k dialogues (seed + generated, exact duplicates skipped). `data/safety_filter.py`
is the final keyword pass over the corpus (it drops 0 here — the generator is clean by
construction, which is exactly the check we want).

### Verification gate

Qualitative: a `<user>` turn yields a `<bot>` reply that is on-topic-enough, in the sweet
child-like voice, and clean. Plus `test_bpe` (round-trip / specials / determinism / save-load)
gates the tokenizer mechanically before any training.

**Result (the run that closed the gate).** `train data/persona.clean.txt --bpe --vocab 1024
--steps 1500 --batch 16 --block 64 --lr 1e-3` on the ~8k-dialogue corpus: BPE saturated at
vocab=1018 (256 bytes + 759 merges + 3 specials), 798k training tokens; loss 6.90 → ~0.27 in
1500 steps (~40 min on CPU, naive loops). `chat` over a dozen varied prompts gives clean,
correctly-formatted, on-voice Moo replies — most on-topic (name, counting, comfort, goodbyes,
storytelling all land), with a few topic mismatches expected of a ~3M-param model at this step
budget (e.g. "favorite animal" can draw the arithmetic reply). Voice consistency and safety are
solid; sharper topic-matching is a "train longer / bigger" lever, not a correctness gap. The
checkpoint round-trips through `save`/`load` and `chat` reads it back via the `.bpe` sidecar.

### Gotchas / decisions

- The space after `<user>`/`<bot>` is part of the format and becomes its own token; the chat
  prompt ends at the `<bot>` token (no trailing space) and lets the model generate the
  leading space, matching how the data was tokenized.
- Char-level (`Tokenizer`) is kept for Phases 0–4; BPE is additive (`--bpe`), not a
  replacement, so the older gates still run unchanged.

---

## Phase 6 — Performance (CPU parallelism)

The stretch phase. Goal: make the fast path faster **without changing what it computes** — the
gate is "same loss/outputs as the verified CPU path." On this machine the RTX 3050 has a driver
but **no CUDA Toolkit (`nvcc`)**, so the CUDA route can't be built/verified here; the CPU route
(the plan's other option, "speed up matmul … on CPU") is what we did: OpenMP across the 16 cores.

### The rule that keeps results identical: parallelize over *owned outputs*, not reductions

A floating-point sum is not associative, so changing the *order* of a reduction changes the last
bits. The safe transform is to give each thread a disjoint set of **output** elements and let it
run the *same* serial reduction for each — every output is written by exactly one thread (no race)
and every dot-product keeps its original summation order (no reassociation). Results are then
independent of thread count. Where each hot loop's output lives decides which axis to parallelize:

- **`matmul` / `linear` (forward):** output element `(i,j)` / row `t` is computed by one
  iteration; parallelize over rows (`i` / `t`). The inner `k`/`in` sum is untouched. Identical.
- **`linear_backward`:** the catch — `dx` is indexed by `t` but `dW`/`dB` by `o` (opposite axes).
  The fused serial loop accumulates both at once, so *neither* axis can be the parallel loop
  without racing the other target. Fix: **split into two passes** —
  - pass 1 `dx[t,i] += Σ_o dy[t,o]·W[o,i]`, parallel over `t` (each `t` owns its `dx` row);
  - pass 2 `dW[o,i] += Σ_t dy[t,o]·x[t,i]`, `dB[o] += Σ_t dy[t,o]`, parallel over `o` (each `o`
    owns its `dW` row + `dB[o]`).

  `dx` still sums over `o` ascending and `dW`/`dB` over `t` ascending, exactly as the fused loop
  did — so the split is bit-identical to serial, and now both passes parallelize race-free. Cost:
  `dy`/`x` are read twice; cheap next to the FLOPs saved.
- **Attention (fwd + bwd):** the `for h` loop. Head `h` reads only its own Q/K/V column slices of
  `qkv` and writes only its own slice of `out_heads` / `dqkv` (offsets `h·d`, `C+h·d`, `2C+h·d`
  never overlap across heads); `scores`/`probs`/`dprobs` are per-iteration locals. So heads are
  independent both ways → parallelize over `h`. (The `linear` calls sit *outside* the `h` loop, so
  no nested parallel regions form; nested OpenMP is off by default anyway.)

Each `#pragma omp parallel for` carries an `if (size > …)` guard so the tiny tensors in the
verification tests stay serial and don't pay fork/join overhead.

### Why it isn't *bit*-identical to the old serial binary (and why that's fine)

The single forward/backward gates still pass at the **same** tolerances as before (Phase 2 logits
max-abs ~6.6e-7 ≤ 1e-4; Phase 3 grads worst-rel ~4.8e-4 ≤ 1e-3). But a 100-step overfit loss
differs in the ~7th digit (`0.0194090` vs `0.0194094`). That drift is **FP contraction**, not a
logic bug: at `-O2` GCC may fuse `a += b*c` into a single `fma` (one rounding instead of two), and
whether it does so depends on how a loop is vectorized/structured — which the restructuring above
changes. It's ~1 ULP per op, chaotically amplified over 100 optimization steps. Re-running the
parallel binary reproduces `0.0194094` **exactly**, which is the real safety check: identical
output run-to-run ⇒ no data race. (To force literal serial/parallel bit-equality one could compile
`-ffp-contract=off`, at a small speed cost; not worth it — the project's contract is tolerance
matching against the PyTorch oracle, which holds.)

### Result & verification

`train … --batch 16 --block 64 --layer 4 --head 4 --embd 128`, 100 overfit steps:
**103.3 s → 28.1 s, ~3.7× on 16 cores.** All five C++ suites green (incl. `test_grad` FD checks
and `test_ops` matmul), the determinism re-run matches to the last digit, and both PyTorch gates
PASS unchanged. Amdahl-limited by attention only reaching `H`-way (4) parallelism while the linears
reach `T`/`o`-way (≥64); pushing attention to per-`i` parallelism is the next lever if needed.

### Build / runtime

`-fopenmp` added to every g++ line and `find_package(OpenMP)` to CMake (both optional — without
it the pragmas no-op to the serial path). libgomp links statically under `-static`, so the exe
stays DLL-free. `OMP_NUM_THREADS` caps threads at run time. **CUDA/GPU remains unbuilt** — it needs
the Toolkit installed; the verified CPU path is the reference any future GPU path must match.

---

## Build environment (this machine)

CMake is **not** installed. Two compilers are present:
- MinGW-w64 g++ 14.2.0 at `C:\msys64\ucrt64\bin\g++.exe` (used for Phase 0 builds).
- MSVC BuildTools 18 (`cl.exe`) under `C:\Program Files (x86)\Microsoft Visual Studio\18`.

`CMakeLists.txt` exists as the intended build, but until CMake is installed, build
directly with g++ (see README / the commands used in Phase 0).
