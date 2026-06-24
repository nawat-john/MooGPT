#include "attention.h"

#include <cmath>
#include <limits>
#include <vector>

#include "ops.h"

namespace moo {

Tensor CausalSelfAttention::forward(const Tensor& x) {
  const int T = x.dim(0);
  const int C = cfg.n_embd;
  const int H = cfg.n_head;
  const int d = cfg.head_dim();
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  const int HT = H * T;

  x_ = x;
  qkv_ = linear(x, c_attn_w, c_attn_b);  // (T, 3C): cols [0,C)=Q, [C,2C)=K, [2C,3C)=V
  out_heads_ = Tensor({T, C});
  probs_.assign(H, Tensor());
  for (int h = 0; h < H; ++h) probs_[h] = Tensor({T, T});  // pre-alloc, zero-filled

  // Phase 6 (per-i): the work for query row i of head h — its scores row, its softmax,
  // and its output row — depends only on the read-only Q/K/V and writes only row i of
  // probs_[h] and row i (head h's columns) of out_heads_. Those outputs are disjoint
  // across every (h,i), so we parallelize over the *flattened* (h,i) space: H*T-way
  // instead of the old H-way, which lets attention use all cores even though H (=4) is
  // smaller than the core count. The softmax is inlined per row (over j<=i) but keeps
  // softmax_rows' exact arithmetic — max over the finite entries, then exp/sum in
  // ascending j; the masked j>i entries that softmax_rows would add as exp(-inf)=0 are
  // simply skipped, so each row is identical and the reduction order is unchanged.
#pragma omp parallel for schedule(static) if (HT > 8)
  for (int hi = 0; hi < HT; ++hi) {
    const int h = hi / T;
    const int i = hi % T;
    const int qoff = h * d;
    const int koff = C + h * d;
    const int voff = 2 * C + h * d;

    const float* qi = qkv_.data() + i * (3 * C) + qoff;
    float* prow = probs_[h].data() + i * T;

    // scores[j] = (q_i . k_j) * scale for j <= i (causal); softmax in place over j<=i.
    float m = -std::numeric_limits<float>::infinity();
    for (int j = 0; j <= i; ++j) {
      const float* kj = qkv_.data() + j * (3 * C) + koff;
      float dot = 0.0f;
      for (int e = 0; e < d; ++e) dot += qi[e] * kj[e];
      const float s = dot * scale;
      prow[j] = s;
      if (s > m) m = s;
    }
    float sum = 0.0f;
    for (int j = 0; j <= i; ++j) {
      const float e = std::exp(prow[j] - m);
      prow[j] = e;
      sum += e;
    }
    const float inv = 1.0f / sum;
    for (int j = 0; j <= i; ++j) prow[j] *= inv;

    // out_i = sum_{j<=i} p[i,j] * v_j  (head h's column slice of out_heads_ row i).
    float* orow = out_heads_.data() + i * C + h * d;
    for (int e = 0; e < d; ++e) orow[e] = 0.0f;
    for (int j = 0; j <= i; ++j) {
      const float p = prow[j];
      const float* vj = qkv_.data() + j * (3 * C) + voff;
      for (int e = 0; e < d; ++e) orow[e] += p * vj[e];
    }
  }

  return linear(out_heads_, c_proj_w, c_proj_b);
}

Tensor CausalSelfAttention::backward(const Tensor& dy) {
  const int T = x_.dim(0);
  const int C = cfg.n_embd;
  const int H = cfg.n_head;
  const int d = cfg.head_dim();
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  const int HT = H * T;

  // Back through the output projection: dy (T,C) -> d_out_heads (T,C).
  Tensor d_out_heads =
      linear_backward(out_heads_, c_proj_w, dy, d_c_proj_w, d_c_proj_b);

  // Gradient w.r.t. the fused QKV (T, 3C), filled by the passes below.
  Tensor dqkv({T, 3 * C});  // zero-filled

  // Per-head dscores (T,T), produced in pass B and consumed by passes C/D.
  std::vector<Tensor> dscores(H);
  for (int h = 0; h < H; ++h) dscores[h] = Tensor({T, T});  // zero-filled

  // Phase 6 (per-i): the serial backward parallelized only over heads (H-way). Going
  // per query/key row hits the same ownership conflict linear_backward has: dq and the
  // softmax JVP are owned by query row i, but dv and dk accumulate over i into key/value
  // row j (opposite axis). So we split into passes, each parallel over the axis (flattened
  // with h) that *owns* its output — every reduction keeps its original ascending order,
  // so the gradients match the serial loop's arithmetic, now at H*T-way parallelism.

  // Pass B: dprobs[i,j] = sum_e d_out_h[i,e] * v[j,e], then softmax-backward of that row
  // -> dscores[h] row i. Owned by (h,i): writes only row i of dscores[h]. (Masked j>i
  // have probs=0, so they stay 0 in dscores and never enter the softmax dot.)
#pragma omp parallel for schedule(static) if (HT > 8)
  for (int hi = 0; hi < HT; ++hi) {
    const int h = hi / T;
    const int i = hi % T;
    const int voff = 2 * C + h * d;
    const float* doh = d_out_heads.data() + i * C + h * d;
    const float* prow = probs_[h].data() + i * T;
    float* dsrow = dscores[h].data() + i * T;

    // dsrow temporarily holds dprobs[j]; accumulate the softmax dot in the same sweep.
    float dot = 0.0f;
    for (int j = 0; j <= i; ++j) {
      const float* vj = qkv_.data() + j * (3 * C) + voff;
      float dp = 0.0f;
      for (int e = 0; e < d; ++e) dp += doh[e] * vj[e];
      dsrow[j] = dp;
      dot += dp * prow[j];
    }
    for (int j = 0; j <= i; ++j) dsrow[j] = prow[j] * (dsrow[j] - dot);
  }

  // Pass A: dv[j,e] += sum_{i>=j} probs[i,j] * d_out_h[i,e]. Owned by (h,j): each owns the
  // value slice of dqkv at key row j. (Independent of dscores; reads only probs/d_out_h.)
#pragma omp parallel for schedule(static) if (HT > 8)
  for (int hj = 0; hj < HT; ++hj) {
    const int h = hj / T;
    const int j = hj % T;
    const int voff = 2 * C + h * d;
    const Tensor& probs = probs_[h];
    float* dvj = dqkv.data() + j * (3 * C) + voff;
    for (int i = j; i < T; ++i) {
      const float p = probs.at(i, j);
      const float* doh = d_out_heads.data() + i * C + h * d;
      for (int e = 0; e < d; ++e) dvj[e] += p * doh[e];
    }
  }

  // scores = (q . k) * scale  =>  d(q.k) = dscores * scale.
  // Pass C: dq[i,e] += scale * sum_{j<=i} dscores[i,j] * k[j,e]. Owned by (h,i).
#pragma omp parallel for schedule(static) if (HT > 8)
  for (int hi = 0; hi < HT; ++hi) {
    const int h = hi / T;
    const int i = hi % T;
    const int qoff = h * d;
    const int koff = C + h * d;
    const float* dsrow = dscores[h].data() + i * T;
    float* dqi = dqkv.data() + i * (3 * C) + qoff;
    for (int j = 0; j <= i; ++j) {
      const float s = dsrow[j] * scale;
      const float* kj = qkv_.data() + j * (3 * C) + koff;
      for (int e = 0; e < d; ++e) dqi[e] += s * kj[e];
    }
  }

  // Pass D: dk[j,e] += scale * sum_{i>=j} dscores[i,j] * q[i,e]. Owned by (h,j).
#pragma omp parallel for schedule(static) if (HT > 8)
  for (int hj = 0; hj < HT; ++hj) {
    const int h = hj / T;
    const int j = hj % T;
    const int qoff = h * d;
    const int koff = C + h * d;
    float* dkj = dqkv.data() + j * (3 * C) + koff;
    for (int i = j; i < T; ++i) {
      const float s = dscores[h].at(i, j) * scale;
      const float* qi = qkv_.data() + i * (3 * C) + qoff;
      for (int e = 0; e < d; ++e) dkj[e] += s * qi[e];
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
