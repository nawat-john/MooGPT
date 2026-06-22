#include "attention.h"

#include <cmath>
#include <limits>

#include "ops.h"

namespace moo {

Tensor CausalSelfAttention::forward(const Tensor& x) {
  const int T = x.dim(0);
  const int C = cfg.n_embd;
  const int H = cfg.n_head;
  const int d = cfg.head_dim();
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));

  x_ = x;
  qkv_ = linear(x, c_attn_w, c_attn_b);  // (T, 3C): cols [0,C)=Q, [C,2C)=K, [2C,3C)=V
  out_heads_ = Tensor({T, C});
  probs_.assign(H, Tensor());

  for (int h = 0; h < H; ++h) {
    const int qoff = h * d;
    const int koff = C + h * d;
    const int voff = 2 * C + h * d;

    Tensor scores({T, T});
    for (int i = 0; i < T; ++i) {
      const float* qi = qkv_.data() + i * (3 * C) + qoff;
      for (int j = 0; j < T; ++j) {
        if (j > i) {
          scores.at(i, j) = -std::numeric_limits<float>::infinity();
          continue;
        }
        const float* kj = qkv_.data() + j * (3 * C) + koff;
        float dot = 0.0f;
        for (int e = 0; e < d; ++e) dot += qi[e] * kj[e];
        scores.at(i, j) = dot * scale;
      }
    }
    Tensor probs = softmax_rows(scores);  // (T, T)

    for (int i = 0; i < T; ++i) {
      float* orow = out_heads_.data() + i * C + h * d;
      for (int e = 0; e < d; ++e) orow[e] = 0.0f;
      for (int j = 0; j <= i; ++j) {
        const float p = probs.at(i, j);
        const float* vj = qkv_.data() + j * (3 * C) + voff;
        for (int e = 0; e < d; ++e) orow[e] += p * vj[e];
      }
    }
    probs_[h] = std::move(probs);
  }

  return linear(out_heads_, c_proj_w, c_proj_b);
}

Tensor CausalSelfAttention::backward(const Tensor& dy) {
  const int T = x_.dim(0);
  const int C = cfg.n_embd;
  const int H = cfg.n_head;
  const int d = cfg.head_dim();
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));

  // Back through the output projection: dy (T,C) -> d_out_heads (T,C).
  Tensor d_out_heads =
      linear_backward(out_heads_, c_proj_w, dy, d_c_proj_w, d_c_proj_b);

  // Gradient w.r.t. the fused QKV (T, 3C), filled per head.
  Tensor dqkv({T, 3 * C});

  for (int h = 0; h < H; ++h) {
    const int qoff = h * d;
    const int koff = C + h * d;
    const int voff = 2 * C + h * d;
    const Tensor& probs = probs_[h];

    // dprobs[i,j] = sum_e d_out_h[i,e] * v[j,e]   (only j <= i contribute)
    // dv[j,e]    += sum_i probs[i,j] * d_out_h[i,e]
    Tensor dprobs({T, T});
    for (int i = 0; i < T; ++i) {
      const float* doh = d_out_heads.data() + i * C + h * d;
      for (int j = 0; j <= i; ++j) {
        const float p = probs.at(i, j);
        const float* vj = qkv_.data() + j * (3 * C) + voff;
        float* dvj = dqkv.data() + j * (3 * C) + voff;
        float dp = 0.0f;
        for (int e = 0; e < d; ++e) {
          dp += doh[e] * vj[e];
          dvj[e] += p * doh[e];
        }
        dprobs.at(i, j) = dp;
      }
    }

    // Softmax backward (per row); masked entries have probs=0 so contribute nothing.
    Tensor dscores = softmax_backward(probs, dprobs);  // (T, T)

    // scores = (q . k) * scale  =>  d(q.k) = dscores * scale.
    //   dq[i,e] += scale * sum_j dscores[i,j] * k[j,e]
    //   dk[j,e] += scale * sum_i dscores[i,j] * q[i,e]
    for (int i = 0; i < T; ++i) {
      const float* qi = qkv_.data() + i * (3 * C) + qoff;
      float* dqi = dqkv.data() + i * (3 * C) + qoff;
      for (int j = 0; j <= i; ++j) {
        const float s = dscores.at(i, j) * scale;
        const float* kj = qkv_.data() + j * (3 * C) + koff;
        float* dkj = dqkv.data() + j * (3 * C) + koff;
        for (int e = 0; e < d; ++e) {
          dqi[e] += s * kj[e];
          dkj[e] += s * qi[e];
        }
      }
    }
  }

  // Back through the fused QKV projection (input was x_).
  return linear_backward(x_, c_attn_w, dqkv, d_c_attn_w, d_c_attn_b);
}

void CausalSelfAttention::zero_grad() {
  d_c_attn_w = Tensor(c_attn_w.shape());
  d_c_attn_b = Tensor(c_attn_b.shape());
  d_c_proj_w = Tensor(c_proj_w.shape());
  d_c_proj_b = Tensor(c_proj_b.shape());
}

}  // namespace moo
