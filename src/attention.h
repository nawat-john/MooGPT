#pragma once

#include "config.h"
#include "tensor.h"

namespace moo {

// Causal multi-head self-attention (forward only, single sequence).
//
// Weights mirror nanoGPT / GPT-2: a single fused c_attn projection produces Q, K, V
// stacked along the feature axis (in that order), and c_proj mixes the concatenated
// head outputs back to n_embd.
struct CausalSelfAttention {
  GPTConfig cfg;
  Tensor c_attn_w;  // (3*C, C)
  Tensor c_attn_b;  // (3*C,)
  Tensor c_proj_w;  // (C, C)
  Tensor c_proj_b;  // (C,)

  // x: (T, C) -> (T, C).
  Tensor forward(const Tensor& x) const;
};

}  // namespace moo
