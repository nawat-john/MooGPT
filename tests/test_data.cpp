// Phase 1 tests: char-level tokenizer round-trip and dataloader batch shapes /
// target-shift correctness. Same dependency-free harness as test_ops.cpp.

#include <cstdio>
#include <random>
#include <string>

#include "dataloader.h"
#include "tokenizer.h"

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

using namespace moo;

// --- Tokenizer ---------------------------------------------------------------

void test_tokenizer_vocab() {
  std::printf("test_tokenizer_vocab\n");
  Tokenizer tok = Tokenizer::from_text("hello world");
  // distinct sorted chars of "hello world": ' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w' = 8
  check(tok.vocab_size() == 8, "vocab_size == 8 (distinct chars)");
  // Sorted by byte value: space (0x20) is first.
  check(tok.chars().front() == ' ', "first char is space (lowest byte)");
  check(tok.encode(" ")[0] == 0, "space maps to id 0");
  check(tok.encode("d")[0] < tok.encode("e")[0], "ids follow sort order");
}

void test_tokenizer_roundtrip() {
  std::printf("test_tokenizer_roundtrip (Phase 1 gate)\n");
  const std::string text =
      "The quick brown fox jumps over the lazy dog.\n"
      "hmm... i think it's pretty! 123 <user> <bot>";
  Tokenizer tok = Tokenizer::from_text(text);

  std::vector<int> ids = tok.encode(text);
  std::string back = tok.decode(ids);
  check(ids.size() == text.size(), "encode length == text length (char-level)");
  check(back == text, "decode(encode(text)) == text");

  // Every id is a valid index into the vocab.
  bool in_range = true;
  for (int id : ids) in_range = in_range && (id >= 0 && id < tok.vocab_size());
  check(in_range, "all ids within [0, vocab_size)");
}

// --- DataLoader --------------------------------------------------------------

void test_dataloader_split() {
  std::printf("test_dataloader_split\n");
  std::vector<int> tokens(100);
  for (int i = 0; i < 100; ++i) tokens[i] = i;
  DataLoader dl(tokens, 0.1f);
  check(dl.train_size() == 90, "train_size == 90 (90%)");
  check(dl.val_size() == 10, "val_size == 10 (10%)");
}

void test_batch_shapes_and_shift() {
  std::printf("test_batch_shapes_and_shift (Phase 1 gate)\n");
  // Token stream = 0,1,2,...,199. Because the value equals the position, we can verify
  // the shift purely from the values: for any drawn window, y[b,t] must equal x[b,t]+1.
  std::vector<int> tokens(200);
  for (int i = 0; i < 200; ++i) tokens[i] = i;
  DataLoader dl(tokens, 0.1f);

  std::mt19937 rng(1234);
  const int B = 4, T = 8;
  Batch batch = dl.get_batch(DataLoader::Split::Train, B, T, rng);

  check(batch.batch_size == B && batch.block_size == T, "batch dims == (B,T)");
  check(batch.x.size() == static_cast<std::size_t>(B) * T, "x has B*T elements");
  check(batch.y.size() == static_cast<std::size_t>(B) * T, "y has B*T elements");

  bool shift_ok = true;
  bool contiguous_ok = true;
  for (int b = 0; b < B; ++b) {
    for (int t = 0; t < T; ++t) {
      // Target is the next token: y[b,t] == x[b,t] + 1 (holds for this 0..N stream).
      if (batch.yat(b, t) != batch.xat(b, t) + 1) shift_ok = false;
      // Within a row, x is a contiguous slice: x[b,t+1] == x[b,t] + 1.
      if (t + 1 < T && batch.xat(b, t + 1) != batch.xat(b, t) + 1) contiguous_ok = false;
    }
  }
  check(shift_ok, "y[b,t] == x[b,t]+1 (target is input shifted by one)");
  check(contiguous_ok, "each row is a contiguous window from the corpus");
}

void test_batch_determinism() {
  std::printf("test_batch_determinism\n");
  std::vector<int> tokens(200);
  for (int i = 0; i < 200; ++i) tokens[i] = i;
  DataLoader dl(tokens, 0.0f);

  std::mt19937 a(42), b(42);
  Batch ba = dl.get_batch(DataLoader::Split::Train, 3, 5, a);
  Batch bb = dl.get_batch(DataLoader::Split::Train, 3, 5, b);
  check(ba.x == bb.x && ba.y == bb.y, "same seed -> same batch");
}

}  // namespace

int main() {
  std::printf("=== MooGPT Phase 1 data tests ===\n");
  test_tokenizer_vocab();
  test_tokenizer_roundtrip();
  test_dataloader_split();
  test_batch_shapes_and_shift();
  test_batch_determinism();

  std::printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
  if (g_failures == 0) {
    std::printf("ALL TESTS PASSED\n");
    return 0;
  }
  std::printf("%d CHECK(S) FAILED\n", g_failures);
  return 1;
}
