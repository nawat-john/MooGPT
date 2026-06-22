#include "block.h"

#include "ops.h"

namespace moo {

Tensor Block::forward(const Tensor& x) {
  x_ = x;
  Tensor a = attn.forward(layernorm(x, ln_1_w, ln_1_b));
  x1_ = add(x, a);  // attention residual
  Tensor m = mlp.forward(layernorm(x1_, ln_2_w, ln_2_b));
  return add(x1_, m);  // mlp residual
}

Tensor Block::backward(const Tensor& dx2) {
  // x2 = x1 + m, with m = mlp(ln_2(x1)).
  Tensor d_ln2 = mlp.backward(dx2);  // dm == dx2 (residual)
  Tensor d_x1_from_ln2 = layernorm_backward(x1_, ln_2_w, d_ln2, d_ln_2_w, d_ln_2_b);
  Tensor dx1 = add(dx2, d_x1_from_ln2);  // x1 flows both directly and through ln_2/mlp

  // x1 = x + a, with a = attn(ln_1(x)).
  Tensor d_ln1 = attn.backward(dx1);  // da == dx1 (residual)
  Tensor d_x_from_ln1 = layernorm_backward(x_, ln_1_w, d_ln1, d_ln_1_w, d_ln_1_b);
  return add(dx1, d_x_from_ln1);  // x flows both directly and through ln_1/attn
}

void Block::zero_grad() {
  d_ln_1_w = Tensor(ln_1_w.shape());
  d_ln_1_b = Tensor(ln_1_b.shape());
  d_ln_2_w = Tensor(ln_2_w.shape());
  d_ln_2_b = Tensor(ln_2_b.shape());
  attn.zero_grad();
  mlp.zero_grad();
}

}  // namespace moo
