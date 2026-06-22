#pragma once

#include <vector>

#include "tensor.h"

namespace moo {

// Cross-entropy over a sequence of logits, averaged across the T positions (matching
// torch.nn.functional.cross_entropy default reduction='mean').
//
// For each position t with logits row z_t and target class y_t:
//   loss_t = -log softmax(z_t)[y_t]
//   loss   = mean_t loss_t
//
// The gradient w.r.t. the logits is the classic softmax-CE result:
//   dlogits[t, :] = (softmax(z_t) - onehot(y_t)) / T
//
// Returns the scalar loss and writes dlogits (shape (T, V)) into the out-param, so the
// caller can kick off backprop without recomputing the softmax.
float cross_entropy(const Tensor& logits, const std::vector<int>& targets,
                    Tensor& dlogits);

}  // namespace moo
