#include "block.h"

#include "ops.h"

namespace moo {

Tensor Block::forward(const Tensor& x) const {
  // Attention sublayer with residual.
  Tensor a = attn.forward(layernorm(x, ln_1_w, ln_1_b));
  Tensor x1 = add(x, a);
  // MLP sublayer with residual.
  Tensor m = mlp.forward(layernorm(x1, ln_2_w, ln_2_b));
  return add(x1, m);
}

}  // namespace moo
