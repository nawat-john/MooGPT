#pragma once

#include <vector>

#include "config.h"
#include "tensor.h"

namespace moo {

// Causal multi-head self-attention (forward + backward, single sequence).
//
// Weights mirror nanoGPT / GPT-2: a single fused c_attn projection produces Q, K, V
// stacked along the feature axis (in that order), and c_proj mixes the concatenated
// head outputs back to n_embd.
struct CausalSelfAttention {
  GPTConfig cfg;
  // Parameters.
  Tensor c_attn_w;  // (3*C, C)
  Tensor c_attn_b;  // (3*C,)
  Tensor c_proj_w;  // (C, C)
  Tensor c_proj_b;  // (C,)
  // Gradients (same shapes).
  Tensor d_c_attn_w, d_c_attn_b, d_c_proj_w, d_c_proj_b;

  // x: (T, C) -> (T, C).
  Tensor forward(const Tensor& x);
  // dy: (T, C) -> dx: (T, C); accumulates parameter gradients.
  Tensor backward(const Tensor& dy);
  void zero_grad();

 private:
  Tensor x_;                  // cached input
  Tensor qkv_;                // (T, 3C) fused projection
  std::vector<Tensor> probs_; // per head, (T, T) attention weights
  Tensor out_heads_;          // (T, C) concatenated head outputs (input to c_proj)
};

}  // namespace moo
