#include "ops.h"

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

}  // namespace moo
