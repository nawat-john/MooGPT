#pragma once

#include "config.h"
#include "tensor.h"

namespace moo {

// Position-wise feed-forward network: Linear(C -> 4C) -> GELU -> Linear(4C -> C).
struct MLP {
  GPTConfig cfg;
  Tensor c_fc_w;    // (4*C, C)
  Tensor c_fc_b;    // (4*C,)
  Tensor c_proj_w;  // (C, 4*C)
  Tensor c_proj_b;  // (C,)

  // x: (T, C) -> (T, C).
  Tensor forward(const Tensor& x) const;
};

}  // namespace moo
