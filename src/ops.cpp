#include "ops.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace moo {

namespace {

void require(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "ops: " << msg << "\n";
    std::abort();
  }
}

}  // namespace

Tensor matmul(const Tensor& a, const Tensor& b) {
  require(a.ndim() == 2 && b.ndim() == 2, "matmul expects 2-D tensors");
  const int M = a.dim(0);
  const int K = a.dim(1);
  require(b.dim(0) == K, "matmul inner dimensions must match (a.cols == b.rows)");
  const int N = b.dim(1);

  Tensor out({M, N});
  const float* ap = a.data();
  const float* bp = b.data();
  float* op = out.data();

  // Naive triple loop. Row-major layout means a[i,k] is at i*K + k and b[k,j] is at
  // k*N + j. The i-k-j order keeps the inner loop striding contiguously over b's row
  // and out's row, which is friendlier to the cache than i-j-k.
  //
  // Phase 6: parallelize over output rows i. Each i owns a distinct row of `out`, so
  // there is no write race, and the inner k-summation order is unchanged — results are
  // bit-identical to the serial loop regardless of thread count. `if(M > 8)` skips the
  // fork/join overhead for the tiny matrices the verification tests use.
#pragma omp parallel for schedule(static) if (M > 8)
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      const float aik = ap[i * K + k];
      const float* brow = bp + k * N;
      float* orow = op + i * N;
      for (int j = 0; j < N; ++j) {
        orow[j] += aik * brow[j];
      }
    }
  }
  return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
  require(a.same_shape(b), "add requires identical shapes");
  Tensor out(a.shape());
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
  return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
  require(a.same_shape(b), "mul requires identical shapes");
  Tensor out(a.shape());
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] * b[i];
  return out;
}

Tensor add_scalar(const Tensor& a, float s) {
  Tensor out(a.shape());
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] + s;
  return out;
}

Tensor mul_scalar(const Tensor& a, float s) {
  Tensor out(a.shape());
  for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] * s;
  return out;
}

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias) {
  require(x.ndim() == 2 && weight.ndim() == 2, "linear expects 2-D x and weight");
  const int T = x.dim(0);
  const int in = x.dim(1);
  const int out = weight.dim(0);
  require(weight.dim(1) == in, "linear: weight in-features must match x cols");
  const bool has_bias = bias.size() != 0;
  require(!has_bias || static_cast<int>(bias.size()) == out,
          "linear: bias size must equal out-features");

  Tensor y({T, out});
  const float* xp = x.data();
  const float* wp = weight.data();  // (out, in), row o is weight for output o
  const float* bp = has_bias ? bias.data() : nullptr;
  float* yp = y.data();

  // Phase 6: parallelize over rows t (one output row per thread, dot-product order
  // unchanged → bit-identical). This is the dominant forward cost (lm_head, c_attn,
  // c_proj, the two MLP linears all route through here).
#pragma omp parallel for schedule(static) if (T > 8)
  for (int t = 0; t < T; ++t) {
    const float* xrow = xp + t * in;
    float* yrow = yp + t * out;
    for (int o = 0; o < out; ++o) {
      const float* wrow = wp + o * in;
      float acc = has_bias ? bp[o] : 0.0f;
      for (int i = 0; i < in; ++i) acc += xrow[i] * wrow[i];
      yrow[o] = acc;
    }
  }
  return y;
}

Tensor gelu(const Tensor& x) {
  Tensor out(x.shape());
  const float inv_sqrt2 = 0.70710678118654752440f;  // 1/sqrt(2)
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float v = x[i];
    out[i] = 0.5f * v * (1.0f + std::erf(v * inv_sqrt2));
  }
  return out;
}

Tensor layernorm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps) {
  require(x.ndim() == 2, "layernorm expects 2-D x (T, C)");
  const int T = x.dim(0);
  const int C = x.dim(1);
  require(static_cast<int>(gamma.size()) == C && static_cast<int>(beta.size()) == C,
          "layernorm: gamma/beta size must equal C");

  Tensor out({T, C});
  for (int t = 0; t < T; ++t) {
    const float* row = x.data() + t * C;
    float* orow = out.data() + t * C;

    // Mean and biased variance over the feature dimension.
    float mean = 0.0f;
    for (int c = 0; c < C; ++c) mean += row[c];
    mean /= C;
    float var = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float d = row[c] - mean;
      var += d * d;
    }
    var /= C;

    const float inv_std = 1.0f / std::sqrt(var + eps);
    for (int c = 0; c < C; ++c) {
      orow[c] = (row[c] - mean) * inv_std * gamma[c] + beta[c];
    }
  }
  return out;
}

Tensor softmax_rows(const Tensor& x) {
  require(x.ndim() == 2, "softmax_rows expects 2-D x");
  const int R = x.dim(0);
  const int N = x.dim(1);
  Tensor out({R, N});
  for (int r = 0; r < R; ++r) {
    const float* row = x.data() + r * N;
    float* orow = out.data() + r * N;
    float m = row[0];
    for (int j = 1; j < N; ++j) m = row[j] > m ? row[j] : m;
    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
      const float e = std::exp(row[j] - m);
      orow[j] = e;
      sum += e;
    }
    const float inv = 1.0f / sum;
    for (int j = 0; j < N; ++j) orow[j] *= inv;
  }
  return out;
}

Tensor linear_backward(const Tensor& x, const Tensor& weight, const Tensor& dy,
                       Tensor& dW, Tensor& dB) {
  const int T = x.dim(0);
  const int in = x.dim(1);
  const int out = weight.dim(0);
  require(dy.dim(0) == T && dy.dim(1) == out, "linear_backward: dy shape mismatch");
  const bool has_bias = dB.size() != 0;

  Tensor dx({T, in});
  // dx = dy @ W ;  dW += dy^T @ x ;  db += colsum(dy)
  //
  // Phase 6: the serial version fused both accumulations in one t-o-i loop, but dx is
  // owned by t while dW/dB are owned by o — opposite axes, so neither can be the parallel
  // loop without a write race on the other target. Splitting into two passes lets each
  // pass parallelize over the axis that owns its output, with no races. Each reduction
  // keeps its original order (dx sums over o ascending; dW/dB sum over t ascending), so
  // the gradients are bit-identical to the fused serial loop.

  // Pass 1: dx[t,i] += sum_o dy[t,o] * W[o,i]  — parallel over t (each t owns dx row t).
#pragma omp parallel for schedule(static) if (T > 8)
  for (int t = 0; t < T; ++t) {
    const float* dyr = dy.data() + t * out;
    float* dxr = dx.data() + t * in;
    for (int o = 0; o < out; ++o) {
      const float g = dyr[o];
      const float* wr = weight.data() + o * in;
      for (int i = 0; i < in; ++i) dxr[i] += g * wr[i];
    }
  }

  // Pass 2: dW[o,i] += sum_t dy[t,o] * x[t,i] ; dB[o] += sum_t dy[t,o]  — parallel over o
  // (each o owns dW row o and dB[o]).
  const float* dyp = dy.data();
  const float* xp = x.data();
#pragma omp parallel for schedule(static) if (out > 8)
  for (int o = 0; o < out; ++o) {
    float* dWr = dW.data() + o * in;
    float db = 0.0f;
    for (int t = 0; t < T; ++t) {
      const float g = dyp[t * out + o];
      const float* xr = xp + t * in;
      for (int i = 0; i < in; ++i) dWr[i] += g * xr[i];
      db += g;
    }
    if (has_bias) dB[o] += db;
  }
  return dx;
}

Tensor gelu_backward(const Tensor& x, const Tensor& dy) {
  Tensor dx(x.shape());
  const float inv_sqrt2 = 0.70710678118654752440f;    // 1/sqrt(2)
  const float inv_sqrt2pi = 0.39894228040143267794f;  // 1/sqrt(2*pi)
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float v = x[i];
    // d/dx [0.5 x (1+erf(x/√2))] = 0.5(1+erf(x/√2)) + x * (1/√(2π)) exp(-x²/2)
    const float cdf = 0.5f * (1.0f + std::erf(v * inv_sqrt2));
    const float pdf = inv_sqrt2pi * std::exp(-0.5f * v * v);
    dx[i] = dy[i] * (cdf + v * pdf);
  }
  return dx;
}

Tensor layernorm_backward(const Tensor& x, const Tensor& gamma, const Tensor& dy,
                          Tensor& dgamma, Tensor& dbeta, float eps) {
  const int T = x.dim(0);
  const int C = x.dim(1);
  Tensor dx({T, C});
  for (int t = 0; t < T; ++t) {
    const float* row = x.data() + t * C;
    const float* dyr = dy.data() + t * C;
    float* dxr = dx.data() + t * C;

    float mean = 0.0f;
    for (int c = 0; c < C; ++c) mean += row[c];
    mean /= C;
    float var = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float d = row[c] - mean;
      var += d * d;
    }
    var /= C;
    const float rstd = 1.0f / std::sqrt(var + eps);

    // xhat = (x - mean) * rstd ; y = gamma*xhat + beta
    // dxhat = dy * gamma ; then the standard normalized-input backprop:
    //   dx = rstd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
    float mean_dxhat = 0.0f, mean_dxhat_xhat = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float xhat = (row[c] - mean) * rstd;
      const float dxhat = dyr[c] * gamma[c];
      mean_dxhat += dxhat;
      mean_dxhat_xhat += dxhat * xhat;
      dgamma[c] += dyr[c] * xhat;  // accumulate param grads
      dbeta[c] += dyr[c];
    }
    mean_dxhat /= C;
    mean_dxhat_xhat /= C;
    for (int c = 0; c < C; ++c) {
      const float xhat = (row[c] - mean) * rstd;
      const float dxhat = dyr[c] * gamma[c];
      dxr[c] = rstd * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat);
    }
  }
  return dx;
}

Tensor softmax_backward(const Tensor& y, const Tensor& dy) {
  const int R = y.dim(0);
  const int N = y.dim(1);
  Tensor dx({R, N});
  for (int r = 0; r < R; ++r) {
    const float* yr = y.data() + r * N;
    const float* dyr = dy.data() + r * N;
    float* dxr = dx.data() + r * N;
    float dot = 0.0f;
    for (int j = 0; j < N; ++j) dot += dyr[j] * yr[j];
    for (int j = 0; j < N; ++j) dxr[j] = yr[j] * (dyr[j] - dot);
  }
  return dx;
}

}  // namespace moo
