#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace moo {

// Byte-level Byte-Pair Encoding tokenizer (Phase 5).
//
// Byte-level means the base alphabet is the 256 possible byte values, so *any* input
// round-trips exactly — there is no <unk> and no character can fall outside the vocab.
// Training repeatedly merges the most frequent adjacent token pair into a new token,
// growing the vocab from 256 up to a target size. To keep merges sensible (and training
// fast) the corpus is first pre-tokenized into "words": maximal runs of one byte class
// (letters / digits / whitespace / other), and merges never cross a word boundary.
//
// Special tokens (e.g. <user>, <bot>, <eot>) are reserved strings given ids *after* the
// byte+merge vocab. They are matched literally during encoding and never split or merged,
// so they always survive as single atomic ids — that is what lets them delimit turns.
class BpeTokenizer {
 public:
  BpeTokenizer() = default;

  // Learn merges from `text` until the vocab reaches `vocab_size` total tokens (which
  // must be >= 256 + specials.size()). Training stops early if the corpus runs out of
  // mergeable pairs. `specials` are reserved, never trained on, never split.
  static BpeTokenizer train(const std::string& text, int vocab_size,
                            const std::vector<std::string>& specials);

  // text -> token ids. Special-token substrings become their reserved id; everything else
  // is BPE-encoded byte-class word by byte-class word.
  std::vector<int> encode(const std::string& text) const;

  // token ids -> text. Inverse of encode (exact for byte/merge tokens; specials decode to
  // their literal string).
  std::string decode(const std::vector<int>& ids) const;

  int vocab_size() const {
    return static_cast<int>(vocab_.size() + id_to_special_.size());
  }
  // Reserved-token id, or -1 if `s` is not a special.
  int special_id(const std::string& s) const;
  bool is_special(int id) const { return id_to_special_.count(id) != 0; }

  void save(const std::string& path) const;
  static BpeTokenizer load(const std::string& path);

 private:
  // id -> raw bytes, for byte tokens (ids 0..255) and merged tokens (256..). Specials are
  // NOT stored here; their ids start at vocab_.size().
  std::vector<std::string> vocab_;
  // Ordered merges: merges_[r] == {a, b} means "pair (a,b) becomes token id 256+r".
  std::vector<std::pair<int, int>> merges_;
  // (a,b) -> rank r, used by encode to apply merges in the order they were learned.
  std::unordered_map<std::uint64_t, int> merge_rank_;
  std::unordered_map<std::string, int> special_to_id_;
  std::unordered_map<int, std::string> id_to_special_;

  static std::uint64_t pair_key(int a, int b) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
           static_cast<std::uint32_t>(b);
  }
  // BPE-encode one word (a list of byte ids) into merged token ids.
  std::vector<int> encode_word(std::vector<int> ids) const;
  // Rebuild vocab_ + merge_rank_ from merges_ (after load()).
  void rebuild_from_merges();
};

}  // namespace moo
