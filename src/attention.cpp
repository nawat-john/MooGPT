#include "attention.h"

#include <cmath>
#include <limits>

#include "ops.h"

namespace moo {

Tensor CausalSelfAttention::forward(const Tensor& x) const {
  const int T = x.dim(0);
  const int C = cfg.n_embd;
  const int H = cfg.n_head;
  const int d = cfg.head_dim();
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));

  // Fused QKV projection: (T, C) -> (T, 3C). Columns [0,C)=Q, [C,2C)=K, [2C,3C)=V.
  Tensor qkv = linear(x, c_attn_w, c_attn_b);

  Tensor out_heads({T, C});  // concatenated per-head outputs

  // Process one head at a time. Head h owns feature columns [h*d, (h+1)*d) within each
  // of the Q/K/V blocks.
  for (int h = 0; h < H; ++h) {
    const int qoff = h * d;          // Q columns for this head
    const int koff = C + h * d;      // K columns
    const int voff = 2 * C + h * d;  // V columns

    // scores[i,j] = (q_i . k_j) * scale, then causal mask (j > i -> -inf), then softmax.
    Tensor scores({T, T});
    for (int i = 0; i < T; ++i) {
      const float* qi = qkv.data() + i * (3 * C) + qoff;
      for (int j = 0; j < T; ++j) {
        if (j > i) {
          scores.at(i, j) = -std::numeric_limits<float>::infinity();
          continue;
        }
        const float* kj = qkv.data() + j * (3 * C) + koff;
        float dot = 0.0f;
        for (int e = 0; e < d; ++e) dot += qi[e] * kj[e];
        scores.at(i, j) = dot * scale;
      }
    }
    Tensor probs = softmax_rows(scores);  // (T, T)

    // out_i = sum_j probs[i,j] * v_j, written into this head's output columns.
    for (int i = 0; i < T; ++i) {
      float* orow = out_heads.data() + i * C + h * d;
      for (int e = 0; e < d; ++e) orow[e] = 0.0f;
      for (int j = 0; j <= i; ++j) {  // causal: only j <= i contribute
        const float p = probs.at(i, j);
        const float* vj = qkv.data() + j * (3 * C) + voff;
        for (int e = 0; e < d; ++e) orow[e] += p * vj[e];
      }
    }
  }

  // Output projection.
  return linear(out_heads, c_proj_w, c_proj_b);
}

}  // namespace moo
