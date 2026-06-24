#include "bpe.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>

namespace moo {

namespace {

void die(const std::string& msg) {
  std::cerr << "BpeTokenizer: " << msg << "\n";
  std::abort();
}

enum class Cls { Space, Alpha, Digit, Other };

Cls classify(unsigned char c) {
  if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f')
    return Cls::Space;
  if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return Cls::Alpha;
  if (c >= '0' && c <= '9') return Cls::Digit;
  return Cls::Other;
}

// Split a plain (special-free) string into words: maximal runs of one byte class.
// "the cat 12!" -> {"the", " ", "cat", " ", "12", "!"}.
std::vector<std::string> split_words(const std::string& s) {
  std::vector<std::string> words;
  std::size_t i = 0;
  while (i < s.size()) {
    const Cls c = classify(static_cast<unsigned char>(s[i]));
    std::size_t j = i + 1;
    while (j < s.size() && classify(static_cast<unsigned char>(s[j])) == c) ++j;
    words.emplace_back(s.substr(i, j - i));
    i = j;
  }
  return words;
}

// One piece of a special-split: either a reserved token (special != "") or plain text.
struct Segment {
  std::string special;  // non-empty => this is a reserved token
  std::string plain;    // used when special is empty
};

// Split text on the longest-matching special token at each position, so e.g. "<user>" is
// pulled out as one segment and the bytes around it stay plain. Longest-match-first means
// overlapping specials resolve deterministically.
std::vector<Segment> split_specials(const std::string& text,
                                    const std::vector<std::string>& specials) {
  std::vector<Segment> segs;
  std::string buf;
  std::size_t i = 0;
  while (i < text.size()) {
    const std::string* hit = nullptr;
    for (const std::string& sp : specials) {
      if (sp.empty()) continue;
      if (text.compare(i, sp.size(), sp) == 0) {
        if (hit == nullptr || sp.size() > hit->size()) hit = &sp;
      }
    }
    if (hit != nullptr) {
      if (!buf.empty()) {
        segs.push_back({"", buf});
        buf.clear();
      }
      segs.push_back({*hit, ""});
      i += hit->size();
    } else {
      buf.push_back(text[i]);
      ++i;
    }
  }
  if (!buf.empty()) segs.push_back({"", buf});
  return segs;
}

std::vector<int> bytes_to_ids(const std::string& w) {
  std::vector<int> ids(w.size());
  for (std::size_t i = 0; i < w.size(); ++i)
    ids[i] = static_cast<unsigned char>(w[i]);
  return ids;
}

}  // namespace

int BpeTokenizer::special_id(const std::string& s) const {
  auto it = special_to_id_.find(s);
  return it == special_to_id_.end() ? -1 : it->second;
}

BpeTokenizer BpeTokenizer::train(const std::string& text, int vocab_size,
                                 const std::vector<std::string>& specials) {
  const int num_specials = static_cast<int>(specials.size());
  if (vocab_size < 256 + num_specials)
    die("vocab_size must be >= 256 + number of special tokens");
  const int num_merges = vocab_size - 256 - num_specials;

  BpeTokenizer t;
  t.vocab_.resize(256);
  for (int b = 0; b < 256; ++b) t.vocab_[b] = std::string(1, static_cast<char>(b));

  // Word frequencies from the plain (special-free) parts of the corpus.
  std::map<std::string, long> word_freq;
  for (const Segment& seg : split_specials(text, specials)) {
    if (!seg.special.empty()) continue;
    for (const std::string& w : split_words(seg.plain)) ++word_freq[w];
  }

  // Working set: each distinct word as a symbol-id list plus its corpus frequency.
  struct Word {
    std::vector<int> syms;
    long freq;
  };
  std::vector<Word> words;
  words.reserve(word_freq.size());
  for (const auto& kv : word_freq) words.push_back({bytes_to_ids(kv.first), kv.second});

  for (int r = 0; r < num_merges; ++r) {
    // Count adjacent pairs across all words, weighted by word frequency.
    std::map<std::pair<int, int>, long> counts;
    for (const Word& w : words)
      for (std::size_t i = 0; i + 1 < w.syms.size(); ++i)
        counts[{w.syms[i], w.syms[i + 1]}] += w.freq;
    if (counts.empty()) break;  // nothing left to merge

    // Most frequent pair; ties broken by the smaller (a,b) for determinism (std::map is
    // ordered, so the first max we meet under > is the lexicographically smallest).
    std::pair<int, int> best{0, 0};
    long best_count = -1;
    for (const auto& kv : counts) {
      if (kv.second > best_count) {
        best_count = kv.second;
        best = kv.first;
      }
    }

    const int new_id = static_cast<int>(t.vocab_.size());  // == 256 + r
    t.vocab_.push_back(t.vocab_[best.first] + t.vocab_[best.second]);
    t.merges_.push_back(best);
    t.merge_rank_[pair_key(best.first, best.second)] = r;

    // Apply the merge in-place to every word.
    for (Word& w : words) {
      std::vector<int> out;
      out.reserve(w.syms.size());
      for (std::size_t i = 0; i < w.syms.size();) {
        if (i + 1 < w.syms.size() && w.syms[i] == best.first &&
            w.syms[i + 1] == best.second) {
          out.push_back(new_id);
          i += 2;
        } else {
          out.push_back(w.syms[i]);
          ++i;
        }
      }
      w.syms.swap(out);
    }
  }

  // Reserve special-token ids right after the byte+merge vocab.
  for (int k = 0; k < num_specials; ++k) {
    const int id = static_cast<int>(t.vocab_.size()) + k;
    t.special_to_id_[specials[k]] = id;
    t.id_to_special_[id] = specials[k];
  }
  return t;
}

std::vector<int> BpeTokenizer::encode_word(std::vector<int> ids) const {
  // Repeatedly merge the present pair with the lowest rank (earliest learned), until none
  // of the remaining adjacent pairs is a known merge.
  while (ids.size() >= 2) {
    int best_rank = std::numeric_limits<int>::max();
    int best_i = -1;
    for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
      auto it = merge_rank_.find(pair_key(ids[i], ids[i + 1]));
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = static_cast<int>(i);
      }
    }
    if (best_i < 0) break;
    const int merged = 256 + best_rank;
    ids[best_i] = merged;
    ids.erase(ids.begin() + best_i + 1);
  }
  return ids;
}

std::vector<int> BpeTokenizer::encode(const std::string& text) const {
  std::vector<std::string> specials;
  specials.reserve(special_to_id_.size());
  for (const auto& kv : special_to_id_) specials.push_back(kv.first);

  std::vector<int> out;
  for (const Segment& seg : split_specials(text, specials)) {
    if (!seg.special.empty()) {
      out.push_back(special_to_id_.at(seg.special));
      continue;
    }
    for (const std::string& w : split_words(seg.plain)) {
      std::vector<int> enc = encode_word(bytes_to_ids(w));
      out.insert(out.end(), enc.begin(), enc.end());
    }
  }
  return out;
}

std::string BpeTokenizer::decode(const std::vector<int>& ids) const {
  std::string out;
  for (int id : ids) {
    auto sp = id_to_special_.find(id);
    if (sp != id_to_special_.end()) {
      out += sp->second;
    } else if (id >= 0 && id < static_cast<int>(vocab_.size())) {
      out += vocab_[id];
    } else {
      die("decode: id out of range: " + std::to_string(id));
    }
  }
  return out;
}

void BpeTokenizer::rebuild_from_merges() {
  vocab_.assign(256, std::string());
  for (int b = 0; b < 256; ++b) vocab_[b] = std::string(1, static_cast<char>(b));
  merge_rank_.clear();
  for (std::size_t r = 0; r < merges_.size(); ++r) {
    const auto& m = merges_[r];
    vocab_.push_back(vocab_[m.first] + vocab_[m.second]);
    merge_rank_[pair_key(m.first, m.second)] = static_cast<int>(r);
  }
}

void BpeTokenizer::save(const std::string& path) const {
  std::ofstream f(path);
  if (!f) die("cannot open file for writing: " + path);
  f << "MOOBPE 1\n";
  f << "merges " << merges_.size() << "\n";
  for (const auto& m : merges_) f << m.first << " " << m.second << "\n";
  // Specials in id order so the file is stable/readable.
  std::vector<std::pair<int, std::string>> sp(id_to_special_.begin(),
                                              id_to_special_.end());
  std::sort(sp.begin(), sp.end());
  f << "specials " << sp.size() << "\n";
  for (const auto& kv : sp) f << kv.second << "\n";
}

BpeTokenizer BpeTokenizer::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) die("cannot open file: " + path);
  std::string tag;
  int version = 0;
  f >> tag >> version;
  if (tag != "MOOBPE" || version != 1) die("bad header (expected 'MOOBPE 1')");

  BpeTokenizer t;
  std::string section;
  std::size_t n = 0;
  f >> section >> n;
  if (section != "merges") die("expected 'merges' section");
  t.merges_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (!(f >> t.merges_[i].first >> t.merges_[i].second)) die("bad merge line");
  }
  t.rebuild_from_merges();

  std::size_t m = 0;
  f >> section >> m;
  if (section != "specials") die("expected 'specials' section");
  for (std::size_t k = 0; k < m; ++k) {
    std::string s;
    if (!(f >> s)) die("bad special line");
    const int id = static_cast<int>(t.vocab_.size() + k);
    t.special_to_id_[s] = id;
    t.id_to_special_[id] = s;
  }
  return t;
}

}  // namespace moo
