#include "mlp.h"

#include "ops.h"

namespace moo {

Tensor MLP::forward(const Tensor& x) const {
  Tensor h = linear(x, c_fc_w, c_fc_b);  // (T, 4C)
  Tensor a = gelu(h);
  return linear(a, c_proj_w, c_proj_b);  // (T, C)
}

}  // namespace moo
