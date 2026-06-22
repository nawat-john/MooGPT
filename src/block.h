#pragma once

#include "attention.h"
#include "config.h"
#include "mlp.h"
#include "tensor.h"

namespace moo {

// One pre-norm transformer block:
//   x1 = x  + attn(ln_1(x))
//   x2 = x1 + mlp (ln_2(x1))
// Pre-norm (LayerNorm before the sublayer, residual around it) keeps the residual path
// an identity at init and is the stable GPT-2 arrangement.
struct Block {
  GPTConfig cfg;
  // Parameters.
  Tensor ln_1_w, ln_1_b;  // (C,)
  CausalSelfAttention attn;
  Tensor ln_2_w, ln_2_b;  // (C,)
  MLP mlp;
  // LayerNorm gradients (attn/mlp own their own grads).
  Tensor d_ln_1_w, d_ln_1_b, d_ln_2_w, d_ln_2_b;

  Tensor forward(const Tensor& x);
  Tensor backward(const Tensor& dx2);
  void zero_grad();

 private:
  Tensor x_, x1_;  // cached: block input, and state after the attention residual
};

}  // namespace moo
