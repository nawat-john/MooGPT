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

}  // namespace moo
