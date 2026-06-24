// Phase 5 unit checks for the byte-level BPE tokenizer. No PyTorch needed: byte-level BPE
// is exactly invertible, so the core property is decode(encode(x)) == x for ANY bytes,
// plus a couple of structural checks (special tokens stay atomic, merges are deterministic
// and frequency-ordered, save/load round-trips).

#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "bpe.h"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(const char* what, bool ok) {
  ++g_checks;
  std::printf(ok ? "  [ok]   %s\n" : "  [FAIL] %s\n", what);
  if (!ok) ++g_failures;
}

const char* kCorpus =
    "the little cat sat on the soft mat.\n"
    "the dog runs fast after the red ball.\n"
    "i like to play in the green grass.\n"
    "the cat and the dog play in the sun.\n"
    "the little dog runs to the little cat.\n";

const std::vector<std::string> kSpecials = {"<user>", "<bot>", "<eot>"};

void test_round_trip() {
  std::printf("test_round_trip\n");
  moo::BpeTokenizer tok = moo::BpeTokenizer::train(kCorpus, 320, kSpecials);

  const std::string samples[] = {
      kCorpus,
      "the cat",
      "",
      "hello, world! 123",
      "<user> why is the sky blue?<eot><bot> it is just pretty!<eot>",
      std::string("\x00\x01\xff binary \xfe", 13),  // arbitrary bytes round-trip too
  };
  bool all = true;
  for (const std::string& s : samples) {
    const std::string back = tok.decode(tok.encode(s));
    if (back != s) {
      std::printf("    mismatch on a sample (len %zu)\n", s.size());
      all = false;
    }
  }
  check("decode(encode(x)) == x for all samples", all);

  // Fuzz: random byte strings must also round-trip.
  std::mt19937 rng(123);
  std::uniform_int_distribution<int> len(0, 64), byte(0, 255);
  bool fuzz_ok = true;
  for (int t = 0; t < 500; ++t) {
    std::string s;
    const int n = len(rng);
    for (int i = 0; i < n; ++i) s.push_back(static_cast<char>(byte(rng)));
    if (tok.decode(tok.encode(s)) != s) fuzz_ok = false;
  }
  check("random-byte fuzz round-trips", fuzz_ok);
}

void test_special_tokens() {
  std::printf("test_special_tokens\n");
  moo::BpeTokenizer tok = moo::BpeTokenizer::train(kCorpus, 320, kSpecials);

  std::vector<int> ids = tok.encode("<user> hi<eot><bot> hello<eot>");
  const bool first_user = !ids.empty() && ids.front() == tok.special_id("<user>");
  const bool last_eot = !ids.empty() && ids.back() == tok.special_id("<eot>");
  check("turn begins with the <user> id", first_user);
  check("turn ends with the <eot> id", last_eot);

  // Each special appears as exactly one atomic id (never split into pieces).
  int user_count = 0, eot_count = 0;
  for (int id : ids) {
    if (id == tok.special_id("<user>")) ++user_count;
    if (id == tok.special_id("<eot>")) ++eot_count;
  }
  check("<user> is one atomic token", user_count == 1);
  check("<eot> appears exactly twice", eot_count == 2);
  check("specials are flagged via is_special", tok.is_special(tok.special_id("<bot>")));
}

void test_deterministic_and_merge_order() {
  std::printf("test_deterministic_and_merge_order\n");
  // "ababab" is one word; pair (a,b) occurs 3x, (b,a) 2x, so the first merge must be a+b.
  moo::BpeTokenizer t1 = moo::BpeTokenizer::train("ababab", 257, {});
  moo::BpeTokenizer t2 = moo::BpeTokenizer::train("ababab", 257, {});
  std::vector<int> e1 = t1.encode("ababab");
  std::vector<int> e2 = t2.encode("ababab");
  check("training is deterministic", e1 == e2);
  // First merge token has id 256; "abab"-style merging makes 'ab' (256) appear.
  bool has_merged = false;
  for (int id : e1)
    if (id == 256) has_merged = true;
  check("most-frequent pair (a,b) was merged into id 256", has_merged);
}

void test_save_load() {
  std::printf("test_save_load\n");
  moo::BpeTokenizer tok = moo::BpeTokenizer::train(kCorpus, 340, kSpecials);
  const std::string path = "build/test_bpe_vocab.txt";
  tok.save(path);
  moo::BpeTokenizer loaded = moo::BpeTokenizer::load(path);

  const std::string s = "<user> the little cat<eot>";
  check("vocab_size survives save/load", loaded.vocab_size() == tok.vocab_size());
  check("encoding identical after save/load", loaded.encode(s) == tok.encode(s));
  check("special ids survive save/load",
        loaded.special_id("<bot>") == tok.special_id("<bot>"));
}

}  // namespace

int main() {
  std::printf("=== MooGPT Phase 5 BPE tokenizer checks ===\n");
  test_round_trip();
  test_special_tokens();
  test_deterministic_and_merge_order();
  test_save_load();

  std::printf("\n%d/%d checks passed\n", g_checks - g_failures, g_checks);
  if (g_failures == 0) {
    std::printf("ALL TESTS PASSED\n");
    return 0;
  }
  std::printf("%d CHECK(S) FAILED\n", g_failures);
  return 1;
}
