#pragma once

#include <vector>

#include "tensor.h"

namespace moo {

// AdamW hyperparameters. Defaults follow the common GPT-2/nanoGPT recipe.
struct AdamWConfig {
  float lr = 3e-4f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.1f;
};

// AdamW optimizer, built from scratch (Phase 4).
//
// Adam keeps two running averages per parameter: the first moment m (mean of the
// gradient) and the second moment v (mean of the gradient squared). Both start at 0,
// so they are biased toward 0 early on; the bias-correction divisors (1 - beta^t) undo
// that. The update is m_hat / (sqrt(v_hat) + eps): a per-parameter adaptive step size.
//
// "W" = decoupled weight decay. Plain Adam folds L2 into the gradient, which then gets
// scaled by the adaptive denominator (so it isn't true weight decay). AdamW instead
// shrinks the parameter directly, p -= lr * wd * p, independent of the moments. Per the
// GPT-2 convention we only decay matrices/embeddings (ndim >= 2), not biases or LayerNorm
// gains/shifts — those 1-D parameters are left undecayed.
class AdamW {
 public:
  // Captures the parameter pointers (their addresses must stay stable for the optimizer's
  // lifetime — GPT::parameters() returns addresses of member tensors, which qualify).
  AdamW(std::vector<Tensor*> params, const AdamWConfig& cfg);

  // One optimization step. `grads` must be parallel to the params passed at construction
  // (same order, same shapes). Advances the step counter used for bias correction, then
  // updates every parameter in place. Fill `grads` via model.zero_grad()+backward() first.
  void step(const std::vector<Tensor*>& grads);

  void set_lr(float lr) { cfg_.lr = lr; }
  float lr() const { return cfg_.lr; }
  int step_count() const { return t_; }

 private:
  AdamWConfig cfg_;
  std::vector<Tensor*> params_;
  std::vector<std::vector<float>> m_;  // first moments, one buffer per parameter
  std::vector<std::vector<float>> v_;  // second moments
  std::vector<bool> decay_;            // per-parameter: apply weight decay? (ndim >= 2)
  int t_ = 0;                          // step count (for bias correction)
};

}  // namespace moo
