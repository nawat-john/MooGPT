#pragma once

#include "tensor.h"

namespace moo {

// Basic tensor ops for Phase 0. Forward-only, fp32, naive loops — correctness and
// clarity first. Backward passes and fused/optimized variants come in later phases.

// Matrix multiply: a (M, K) @ b (K, N) -> (M, N). 2-D only for now.
Tensor matmul(const Tensor& a, const Tensor& b);

// Elementwise add: a + b. Shapes must match exactly (no broadcasting yet).
Tensor add(const Tensor& a, const Tensor& b);

// Elementwise multiply (Hadamard): a * b. Shapes must match exactly.
Tensor mul(const Tensor& a, const Tensor& b);

// Scalar operations.
Tensor add_scalar(const Tensor& a, float s);
Tensor mul_scalar(const Tensor& a, float s);

// --- Phase 2: forward-only neural-net ops ------------------------------------
// These operate on 2-D tensors with the "row = token/position, col = feature" layout.

// Linear layer: x (T, in) @ W^T + b -> (T, out).
// W has shape (out, in) and b has shape (out,) — matching PyTorch nn.Linear's storage,
// so weights exported from the reference drop straight in. Pass an empty bias tensor
// (size 0) to skip the bias (used by the weight-untied lm_head).
Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias);

// GELU activation, exact (erf) form: 0.5 * x * (1 + erf(x / sqrt(2))).
// Must match the reference's gelu exactly, so we fix the erf variant on both sides.
Tensor gelu(const Tensor& x);

// LayerNorm over the last dimension of x (T, C). gamma/beta have shape (C,).
// Uses the biased variance (divide by C), matching torch.nn.LayerNorm.
Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                 float eps = 1e-5f);

// Row-wise softmax over the last dimension of a 2-D tensor (numerically stabilized by
// subtracting the row max).
Tensor softmax_rows(const Tensor& x);

}  // namespace moo
