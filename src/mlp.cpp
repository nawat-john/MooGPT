#include "mlp.h"

#include "ops.h"

namespace moo {

Tensor MLP::forward(const Tensor& x) {
  x_ = x;
  fc_out_ = linear(x, c_fc_w, c_fc_b);  // (T, 4C)
  act_ = gelu(fc_out_);
  return linear(act_, c_proj_w, c_proj_b);  // (T, C)
}

Tensor MLP::backward(const Tensor& dy) {
  // Through the output projection (input was act_).
  Tensor d_act = linear_backward(act_, c_proj_w, dy, d_c_proj_w, d_c_proj_b);
  // Through GELU (input was fc_out_).
  Tensor d_fc = gelu_backward(fc_out_, d_act);
  // Through the input projection (input was x_).
  return linear_backward(x_, c_fc_w, d_fc, d_c_fc_w, d_c_fc_b);
}

void MLP::zero_grad() {
  d_c_fc_w = Tensor(c_fc_w.shape());
  d_c_fc_b = Tensor(c_fc_b.shape());
  d_c_proj_w = Tensor(c_proj_w.shape());
  d_c_proj_b = Tensor(c_proj_b.shape());
}

}  // namespace moo
