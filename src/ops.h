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

// --- Phase 3: backward passes ------------------------------------------------
// Each helper takes the upstream gradient dy (same shape as the op's output) and the
// inputs needed to evaluate the local Jacobian, returns the input gradient, and
// ACCUMULATES (+=) into the parameter-gradient buffers. Accumulation (not assignment)
// lets a parameter used in several places sum its contributions; callers zero grads first.

// Backward of linear (y = x @ W^T + b).
//   dx (T, in) = dy @ W
//   dW (out, in) += dy^T @ x
//   db (out,)    += sum over rows of dy   (pass empty dB to skip, for the bias-less head)
Tensor linear_backward(const Tensor& x, const Tensor& weight, const Tensor& dy,
                       Tensor& dW, Tensor& dB);

// Backward of exact-erf GELU. Recomputes the local derivative from the original input x.
Tensor gelu_backward(const Tensor& x, const Tensor& dy);

// Backward of LayerNorm over the last dim. Recomputes mean/variance from x.
//   accumulates dgamma += sum_t dy*xhat, dbeta += sum_t dy.
Tensor layernorm_backward(const Tensor& x, const Tensor& gamma, const Tensor& dy,
                          Tensor& dgamma, Tensor& dbeta, float eps = 1e-5f);

// Backward of row-wise softmax, given the softmax OUTPUT y (not the input) and dy.
//   dx[i] = y[i] * (dy[i] - sum_j dy[j] y[j])   per row.
Tensor softmax_backward(const Tensor& y, const Tensor& dy);

}  // namespace moo
