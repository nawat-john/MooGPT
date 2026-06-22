#pragma once

#include "config.h"
#include "tensor.h"

namespace moo {

// Position-wise feed-forward network: Linear(C -> 4C) -> GELU -> Linear(4C -> C).
//
// forward() caches the activations backward() needs; call zero_grad() before a
// backward pass (it allocates/zeros the gradient buffers).
struct MLP {
  GPTConfig cfg;
  // Parameters.
  Tensor c_fc_w;    // (4*C, C)
  Tensor c_fc_b;    // (4*C,)
  Tensor c_proj_w;  // (C, 4*C)
  Tensor c_proj_b;  // (C,)
  // Gradients (same shapes as the parameters above).
  Tensor d_c_fc_w, d_c_fc_b, d_c_proj_w, d_c_proj_b;

  // x: (T, C) -> (T, C).
  Tensor forward(const Tensor& x);
  // dy: (T, C) -> dx: (T, C); accumulates parameter gradients.
  Tensor backward(const Tensor& dy);
  void zero_grad();

 private:
  Tensor x_, fc_out_, act_;  // cached: input, pre-GELU, post-GELU
};

}  // namespace moo
