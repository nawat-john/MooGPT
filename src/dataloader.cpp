#include "dataloader.h"

#include <cstdlib>
#include <iostream>

namespace moo {

DataLoader::DataLoader(std::vector<int> tokens, float val_fraction) {
  if (val_fraction < 0.0f || val_fraction >= 1.0f) {
    std::cerr << "DataLoader: val_fraction must be in [0, 1), got " << val_fraction << "\n";
    std::abort();
  }
  const std::size_t n = tokens.size();
  // Train is the leading (1 - val_fraction); val is the trailing tail. Splitting by a
  // contiguous cut (rather than interleaving) keeps val genuinely held-out text.
  const std::size_t n_val = static_cast<std::size_t>(static_cast<double>(n) * val_fraction);
  const std::size_t n_train = n - n_val;
  train_.assign(tokens.begin(), tokens.begin() + n_train);
  val_.assign(tokens.begin() + n_train, tokens.end());
}

const std::vector<int>& DataLoader::data_for(Split split) const {
  return split == Split::Train ? train_ : val_;
}

Batch DataLoader::get_batch(Split split, int batch_size, int block_size,
                            std::mt19937& rng) const {
  const std::vector<int>& data = data_for(split);
  // Need block_size+1 tokens per window (x is the first T, y is shifted by one).
  if (static_cast<std::size_t>(block_size) + 1 > data.size()) {
    std::cerr << "DataLoader::get_batch: split has " << data.size()
              << " tokens, needs at least block_size+1 = " << (block_size + 1) << "\n";
    std::abort();
  }

  Batch batch;
  batch.batch_size = batch_size;
  batch.block_size = block_size;
  batch.x.resize(static_cast<std::size_t>(batch_size) * block_size);
  batch.y.resize(static_cast<std::size_t>(batch_size) * block_size);

  // Valid start indices are [0, data.size() - block_size - 1] so that s + block_size
  // (the last target) stays in range.
  std::uniform_int_distribution<std::size_t> pick(0, data.size() - block_size - 1);
  for (int b = 0; b < batch_size; ++b) {
    const std::size_t s = pick(rng);
    for (int t = 0; t < block_size; ++t) {
      batch.x[b * block_size + t] = data[s + t];
      batch.y[b * block_size + t] = data[s + t + 1];  // shifted by one
    }
  }
  return batch;
}

}  // namespace moo
