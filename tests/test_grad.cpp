// Phase 3 gate, part 1: finite-difference gradient checks in pure C++.
//
// For each op we build a scalar loss L = sum(g ⊙ op(inputs)) for a fixed random upstream
// gradient g. Then dL/dinput == op_backward(g), which we compare against central-
// difference estimates  (L(x+eps) - L(x-eps)) / (2 eps).  fp32 makes FD noisy, so the
// tolerance is loose (this catches sign/structure/factor bugs; the PyTorch autograd
// check provides the tight end-to-end verification).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <string>

#include "loss.h"
#include "ops.h"
#include "tensor.h"

namespace {

int g_failures = 0;
int g_checks = 0;
std::mt19937 g_rng(7);

float urand() {
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  return d(g_rng);
}

void fill_rand(moo::Tensor& t) {
  for (std::size_t i = 0; i < t.size(); ++i) t[i] = urand();
}

using moo::Tensor;

// Scalar loss L = sum(g ⊙ out).
double weighted_sum(const Tensor& out, const Tensor& g) {
  double s = 0.0;
  for (std::size_t i = 0; i < out.size(); ++i) s += static_cast<double>(out[i]) * g[i];
  return s;
}

// Compare an analytic gradient tensor against finite differences of `loss` w.r.t. each
// element of `param`. eps and tolerances tuned for fp32 FD noise.
void check_fd(const std::string& what, Tensor& param, const Tensor& analytic,
              const std::function<double()>& loss, float eps = 1e-2f,
              float rtol = 3e-2f, float atol = 3e-3f) {
  ++g_checks;
  float worst = 0.0f;
  for (std::size_t i = 0; i < param.size(); ++i) {
    const float orig = param[i];
    param[i] = orig + eps;
    const double lp = loss();
    param[i] = orig - eps;
    const double lm = loss();
    param[i] = orig;
    const float num = static_cast<float>((lp - lm) / (2.0 * eps));
    const float a = analytic[i];
    const float err = std::fabs(a - num);
    if (err > atol + rtol * std::fabs(num)) worst = std::max(worst, err);
  }
  if (worst == 0.0f) {
    std::printf("  [ok]   %s\n", what.c_str());
  } else {
    ++g_failures;
    std::printf("  [FAIL] %s (worst abs err %.4e)\n", what.c_str(), worst);
  }
}

void test_linear() {
  std::printf("test_linear backward\n");
  const int T = 3, in = 4, out = 5;
  Tensor x({T, in}), W({out, in}), b({out}), g({T, out});
  fill_rand(x); fill_rand(W); fill_rand(b); fill_rand(g);

  auto loss = [&]() { return weighted_sum(moo::linear(x, W, b), g); };

  Tensor dW(W.shape()), dB(b.shape());
  Tensor dx = moo::linear_backward(x, W, g, dW, dB);
  check_fd("dx", x, dx, loss);
  check_fd("dW", W, dW, loss);
  check_fd("db", b, dB, loss);
}

void test_gelu() {
  std::printf("test_gelu backward\n");
  Tensor x({16}), g({16});
  fill_rand(x); fill_rand(g);
  auto loss = [&]() { return weighted_sum(moo::gelu(x), g); };
  Tensor dx = moo::gelu_backward(x, g);
  check_fd("dx", x, dx, loss);
}

void test_layernorm() {
  std::printf("test_layernorm backward\n");
  const int T = 3, C = 6;
  Tensor x({T, C}), gamma({C}), beta({C}), g({T, C});
  fill_rand(x); fill_rand(gamma); fill_rand(beta); fill_rand(g);
  auto loss = [&]() { return weighted_sum(moo::layernorm(x, gamma, beta), g); };

  Tensor dgamma(gamma.shape()), dbeta(beta.shape());
  Tensor dx = moo::layernorm_backward(x, gamma, g, dgamma, dbeta);
  check_fd("dx", x, dx, loss);
  check_fd("dgamma", gamma, dgamma, loss);
  check_fd("dbeta", beta, dbeta, loss);
}

void test_softmax() {
  std::printf("test_softmax backward\n");
  const int R = 3, N = 5;
  Tensor x({R, N}), g({R, N});
  fill_rand(x); fill_rand(g);
  auto loss = [&]() { return weighted_sum(moo::softmax_rows(x), g); };
  Tensor y = moo::softmax_rows(x);
  Tensor dx = moo::softmax_backward(y, g);
  check_fd("dx", x, dx, loss);
}

void test_cross_entropy() {
  std::printf("test_cross_entropy backward\n");
  const int T = 4, V = 7;
  Tensor logits({T, V});
  fill_rand(logits);
  std::vector<int> targets = {0, 3, 6, 2};

  Tensor dlogits;
  moo::cross_entropy(logits, targets, dlogits);  // analytic grad in dlogits

  auto loss = [&]() {
    Tensor scratch;
    return static_cast<double>(moo::cross_entropy(logits, targets, scratch));
  };
  check_fd("dlogits", logits, dlogits, loss);
}

}  // namespace

int main() {
  std::printf("=== MooGPT Phase 3 finite-difference gradient checks ===\n");
  test_linear();
  test_gelu();
  test_layernorm();
  test_softmax();
  test_cross_entropy();

  std::printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
  if (g_failures == 0) {
    std::printf("ALL TESTS PASSED\n");
    return 0;
  }
  std::printf("%d CHECK(S) FAILED\n", g_failures);
  return 1;
}
