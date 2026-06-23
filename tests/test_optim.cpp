// Phase 4 unit checks for AdamW. No PyTorch needed — we test the optimizer against
// problems with known answers:
//   1. Convex minimization: f(x) = sum (x - target)^2, grad = 2(x - target). AdamW must
//      drive x to target (this exercises moments, bias correction, the adaptive step).
//   2. Decoupled weight decay: with zero gradient, a matrix parameter must shrink
//      geometrically toward 0 (p *= 1 - lr*wd each step).
//   3. Selective decay: a 1-D parameter (bias/LayerNorm-shaped) is NOT decayed, so with
//      zero gradient it stays put.

#include <cmath>
#include <cstdio>
#include <vector>

#include "optimizer.h"
#include "tensor.h"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(const char* what, bool ok, double detail = 0.0) {
  ++g_checks;
  if (ok) {
    std::printf("  [ok]   %s\n", what);
  } else {
    ++g_failures;
    std::printf("  [FAIL] %s (%.6e)\n", what, detail);
  }
}

using moo::Tensor;

void test_convex_minimization() {
  std::printf("test_convex_minimization\n");
  Tensor p({3, 2});
  p.set({0, 0, 0, 0, 0, 0});
  const float target[6] = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 2.0f};

  moo::AdamWConfig cfg;
  cfg.lr = 0.05f;
  cfg.weight_decay = 0.0f;  // pure minimization: the optimum is exactly `target`
  moo::AdamW opt({&p}, cfg);

  Tensor grad(p.shape());
  std::vector<Tensor*> grads = {&grad};
  for (int step = 0; step < 5000; ++step) {
    for (std::size_t i = 0; i < p.size(); ++i) grad[i] = 2.0f * (p[i] - target[i]);
    opt.step(grads);
  }

  float worst = 0.0f;
  for (std::size_t i = 0; i < p.size(); ++i)
    worst = std::max(worst, std::fabs(p[i] - target[i]));
  check("x converges to target", worst < 1e-2f, worst);
}

void test_decoupled_weight_decay() {
  std::printf("test_decoupled_weight_decay\n");
  Tensor p({2, 2});  // ndim >= 2 -> decayed
  p.fill(1.0f);

  moo::AdamWConfig cfg;
  cfg.lr = 0.1f;
  cfg.weight_decay = 0.5f;  // shrink factor per step = 1 - lr*wd = 0.95
  moo::AdamW opt({&p}, cfg);

  Tensor grad(p.shape());  // zero gradient: only the decay term acts
  std::vector<Tensor*> grads = {&grad};
  const int steps = 100;
  for (int step = 0; step < steps; ++step) opt.step(grads);

  const float expected = std::pow(1.0f - cfg.lr * cfg.weight_decay, static_cast<float>(steps));
  float worst = 0.0f;
  for (std::size_t i = 0; i < p.size(); ++i) worst = std::max(worst, std::fabs(p[i] - expected));
  check("matrix decays geometrically with zero grad", worst < 1e-3f, worst);
}

void test_no_decay_on_1d() {
  std::printf("test_no_decay_on_1d\n");
  Tensor p({4});  // ndim < 2 -> NOT decayed (bias / LayerNorm convention)
  p.fill(1.0f);

  moo::AdamWConfig cfg;
  cfg.lr = 0.1f;
  cfg.weight_decay = 0.5f;
  moo::AdamW opt({&p}, cfg);

  Tensor grad(p.shape());  // zero gradient and no decay -> parameter is untouched
  std::vector<Tensor*> grads = {&grad};
  for (int step = 0; step < 100; ++step) opt.step(grads);

  float worst = 0.0f;
  for (std::size_t i = 0; i < p.size(); ++i) worst = std::max(worst, std::fabs(p[i] - 1.0f));
  check("1-D parameter is not weight-decayed", worst < 1e-6f, worst);
}

}  // namespace

int main() {
  std::printf("=== MooGPT Phase 4 AdamW checks ===\n");
  test_convex_minimization();
  test_decoupled_weight_decay();
  test_no_decay_on_1d();

  std::printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
  if (g_failures == 0) {
    std::printf("ALL TESTS PASSED\n");
    return 0;
  }
  std::printf("%d CHECK(S) FAILED\n", g_failures);
  return 1;
}
