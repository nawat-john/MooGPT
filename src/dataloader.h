#pragma once

#include <cstddef>
#include <random>
#include <vector>

namespace moo {

// One training batch of token ids, row-major (batch_size, block_size).
//
// For language modeling the target is the input shifted left by one: the model reads
// x[b, 0..t] and must predict the next token, which is y[b, t] = x[b, t+1]. So a single
// sampled window of length block_size+1 from the corpus yields both x (first T tokens)
// and y (last T tokens).
struct Batch {
  int batch_size = 0;
  int block_size = 0;
  std::vector<int> x;  // size batch_size * block_size
  std::vector<int> y;  // size batch_size * block_size

  int xat(int b, int t) const { return x[b * block_size + t]; }
  int yat(int b, int t) const { return y[b * block_size + t]; }
};

// Holds a tokenized corpus, split into train/val contiguous halves, and serves random
// fixed-length batches. The corpus streams from a flat std::vector<int> in memory; for
// big corpora later this can be swapped for a memory-mapped file without changing the
// batch interface.
class DataLoader {
 public:
  enum class Split { Train, Val };

  // val_fraction is the tail fraction of tokens reserved for validation (default 10%).
  explicit DataLoader(std::vector<int> tokens, float val_fraction = 0.1f);

  // Draw batch_size independent random windows of length block_size from the given split.
  // The rng is passed in so tests/training control determinism (seeded mt19937).
  Batch get_batch(Split split, int batch_size, int block_size, std::mt19937& rng) const;

  std::size_t train_size() const { return train_.size(); }
  std::size_t val_size() const { return val_.size(); }

 private:
  const std::vector<int>& data_for(Split split) const;

  std::vector<int> train_;
  std::vector<int> val_;
};

}  // namespace moo
