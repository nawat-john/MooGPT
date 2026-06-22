#include "tokenizer.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>

namespace moo {

Tokenizer Tokenizer::from_text(const std::string& text) {
  // std::set<char> gives us distinct chars already sorted by value.
  std::set<char> distinct(text.begin(), text.end());
  std::vector<char> chars(distinct.begin(), distinct.end());
  return from_chars(chars);
}

Tokenizer Tokenizer::from_chars(const std::vector<char>& chars) {
  Tokenizer t;
  t.itos_ = chars;
  std::sort(t.itos_.begin(), t.itos_.end());
  t.itos_.erase(std::unique(t.itos_.begin(), t.itos_.end()), t.itos_.end());
  for (int i = 0; i < static_cast<int>(t.itos_.size()); ++i) {
    t.stoi_[t.itos_[i]] = i;
  }
  return t;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
  std::vector<int> ids;
  ids.reserve(text.size());
  for (char c : text) {
    auto it = stoi_.find(c);
    if (it == stoi_.end()) {
      std::cerr << "Tokenizer::encode: character not in vocab: '" << c << "' (byte "
                << static_cast<int>(static_cast<unsigned char>(c)) << ")\n";
      std::abort();
    }
    ids.push_back(it->second);
  }
  return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string out;
  out.reserve(ids.size());
  for (int id : ids) {
    if (id < 0 || id >= vocab_size()) {
      std::cerr << "Tokenizer::decode: id out of range: " << id << " (vocab size "
                << vocab_size() << ")\n";
      std::abort();
    }
    out.push_back(itos_[id]);
  }
  return out;
}

}  // namespace moo
