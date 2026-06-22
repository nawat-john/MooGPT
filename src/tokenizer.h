#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace moo {

// Char-level tokenizer (Phase 1). The vocabulary is the set of distinct characters in
// the training corpus, sorted by byte value so ids are deterministic across runs. Each
// id is just the index of that character in the sorted vocab.
//
// Phase 5 will add a BPE tokenizer; this one keeps a whole moving part out of the way
// while the model/training pipeline is brought up.
class Tokenizer {
 public:
  Tokenizer() = default;

  // Build the vocab from a corpus: collect distinct chars, sort, assign ids 0..V-1.
  static Tokenizer from_text(const std::string& text);

  // Build directly from an explicit, already-ordered character set (e.g. loaded vocab).
  static Tokenizer from_chars(const std::vector<char>& chars);

  int vocab_size() const { return static_cast<int>(itos_.size()); }
  const std::vector<char>& chars() const { return itos_; }

  // text -> token ids. Aborts on a character not in the vocab (loud failure while
  // learning beats silently dropping or guessing).
  std::vector<int> encode(const std::string& text) const;

  // token ids -> text.
  std::string decode(const std::vector<int>& ids) const;

  bool contains(char c) const { return stoi_.find(c) != stoi_.end(); }

 private:
  std::vector<char> itos_;             // id   -> char
  std::unordered_map<char, int> stoi_; // char -> id
};

}  // namespace moo
