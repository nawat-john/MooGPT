// Phase 0 unit tests. Deliberately dependency-free (no gtest/Catch) — a tiny harness
// keeps the build trivial and the failures obvious, in the spirit of "understand every
// moving part." Returns non-zero exit code if any check fails so ctest/CI can gate on it.

#include <cmath>
#include <cstdio>
#include <string>

#include "ops.h"
#include "tensor.h"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool cond, const std::string& what) {
  ++g_checks;
  if (cond) {
    std::printf("  [ok]   %s\n", what.c_str());
  } else {
    ++g_failures;
    std::printf("  [FAIL] %s\n", what.c_str());
  }
}

void check_close(float got, float want, const std::string& what, float atol = 1e-5f) {
  check(std::fabs(got - want) <= atol, what + " (got " + std::to_string(got) +
                                           ", want " + std::to_string(want) + ")");
}

using namespace moo;

// --- Tensor basics -----------------------------------------------------------

void test_tensor_layout() {
  std::printf("test_tensor_layout\n");
  Tensor t({2, 3});
  check(t.ndim() == 2, "ndim == 2");
  check(t.size() == 6, "size == 6");
  // Row-major strides for (2,3) are (3,1).
  check(t.strides()[0] == 3 && t.strides()[1] == 1, "row-major strides == (3,1)");
  check(t[0] == 0.0f, "zero-initialized");

  // (i,j) maps to flat i*cols + j.
  check(t.flat_index({0, 0}) == 0, "flat(0,0) == 0");
  check(t.flat_index({0, 2}) == 2, "flat(0,2) == 2");
  check(t.flat_index({1, 0}) == 3, "flat(1,0) == 3");
  check(t.flat_index({1, 2}) == 5, "flat(1,2) == 5");

  t.at(1, 2) = 9.0f;
  check(t[5] == 9.0f, "at(1,2) writes flat index 5");
}

void test_tensor_3d_strides() {
  std::printf("test_tensor_3d_strides\n");
  // (2,3,4): strides (12,4,1).
  Tensor t({2, 3, 4});
  check(t.size() == 24, "size == 24");
  check(t.strides()[0] == 12 && t.strides()[1] == 4 && t.strides()[2] == 1,
        "strides == (12,4,1)");
  check(t.flat_index({1, 2, 3}) == 1 * 12 + 2 * 4 + 3, "flat(1,2,3) == 23");
}

// --- matmul: the Phase 0 gate ------------------------------------------------

void test_matmul_hand_checked() {
  std::printf("test_matmul_hand_checked (Phase 0 gate)\n");
  // A is 2x3, B is 3x2.
  //   A = [[1,2,3],
  //        [4,5,6]]
  //   B = [[7, 8],
  //        [9,10],
  //        [11,12]]
  // C = A@B (2x2), computed by hand:
  //   C[0,0] = 1*7 + 2*9  + 3*11 = 7 + 18 + 33 = 58
  //   C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
  //   C[1,0] = 4*7 + 5*9  + 6*11 = 28 + 45 + 66 = 139
  //   C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
  Tensor a({2, 3});
  a.set({1, 2, 3, 4, 5, 6});
  Tensor b({3, 2});
  b.set({7, 8, 9, 10, 11, 12});

  Tensor c = matmul(a, b);
  check(c.shape().size() == 2 && c.dim(0) == 2 && c.dim(1) == 2, "shape == (2,2)");
  check_close(c.at(0, 0), 58.0f, "C[0,0] == 58");
  check_close(c.at(0, 1), 64.0f, "C[0,1] == 64");
  check_close(c.at(1, 0), 139.0f, "C[1,0] == 139");
  check_close(c.at(1, 1), 154.0f, "C[1,1] == 154");
}

void test_matmul_identity() {
  std::printf("test_matmul_identity\n");
  Tensor a({2, 2});
  a.set({3, 5, 7, 9});
  Tensor id({2, 2});
  id.set({1, 0, 0, 1});
  check(matmul(a, id).allclose(a), "A @ I == A");
  check(matmul(id, a).allclose(a), "I @ A == A");
}

// --- elementwise -------------------------------------------------------------

void test_elementwise() {
  std::printf("test_elementwise\n");
  Tensor a({2, 2});
  a.set({1, 2, 3, 4});
  Tensor b({2, 2});
  b.set({10, 20, 30, 40});

  Tensor s = add(a, b);
  Tensor expect_s({2, 2});
  expect_s.set({11, 22, 33, 44});
  check(s.allclose(expect_s), "add elementwise");

  Tensor p = mul(a, b);
  Tensor expect_p({2, 2});
  expect_p.set({10, 40, 90, 160});
  check(p.allclose(expect_p), "mul elementwise (Hadamard)");

  Tensor sc = mul_scalar(a, 2.0f);
  Tensor expect_sc({2, 2});
  expect_sc.set({2, 4, 6, 8});
  check(sc.allclose(expect_sc), "mul_scalar by 2");

  Tensor as = add_scalar(a, 100.0f);
  Tensor expect_as({2, 2});
  expect_as.set({101, 102, 103, 104});
  check(as.allclose(expect_as), "add_scalar by 100");
}

}  // namespace

int main() {
  std::printf("=== MooGPT Phase 0 ops tests ===\n");
  test_tensor_layout();
  test_tensor_3d_strides();
  test_matmul_hand_checked();
  test_matmul_identity();
  test_elementwise();

  std::printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
  if (g_failures == 0) {
    std::printf("ALL TESTS PASSED\n");
    return 0;
  }
  std::printf("%d CHECK(S) FAILED\n", g_failures);
  return 1;
}
