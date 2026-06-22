#pragma once

#include "attention.h"
#include "config.h"
#include "mlp.h"
#include "tensor.h"

namespace moo {

// One pre-norm transformer block:
//   x = x + attn(ln_1(x))
//   x = x + mlp (ln_2(x))
// Pre-norm (LayerNorm before the sublayer, residual around it) is the stable GPT-2
// arrangement and keeps the residual path an identity at init.
struct Block {
  GPTConfig cfg;
  Tensor ln_1_w, ln_1_b;  // (C,)
  CausalSelfAttention attn;
  Tensor ln_2_w, ln_2_b;  // (C,)
  MLP mlp;

  Tensor forward(const Tensor& x) const;
};

}  // namespace moo
