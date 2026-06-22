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

}  // namespace moo
