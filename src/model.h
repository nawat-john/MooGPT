#pragma once

#include <random>
#include <string>
#include <vector>

#include "block.h"
#include "config.h"
#include "tensor.h"

namespace moo {

// The full GPT: embeddings -> N transformer blocks -> final LayerNorm -> lm_head.
//
// The lm_head is an independent (untied) weight matrix here, not shared with the token
// embedding. Tying is the GPT-2 default and saves parameters, but keeping it separate
// makes the forward and the weight export trivially explicit while bringing the pipeline
// up. Revisit tying once correctness is locked in.
class GPT {
 public:
  GPTConfig cfg;

  // Parameters.
  Tensor wte;        // (vocab, C) token embedding
  Tensor wpe;        // (block_size, C) positional embedding
  std::vector<Block> blocks;
  Tensor ln_f_w, ln_f_b;  // (C,)
  Tensor lm_head_w;       // (vocab, C), no bias

  // Forward pass on a single sequence of token ids (length T <= block_size).
  // Returns logits of shape (T, vocab_size).
  Tensor forward(const std::vector<int>& tokens) const;

  // Autoregressive sampling. Starts from `prompt`, appends `max_new_tokens` sampled ids.
  // temperature > 0 scales logits; top_k <= 0 disables top-k filtering.
  std::vector<int> generate(const std::vector<int>& prompt, int max_new_tokens,
                            float temperature, int top_k, std::mt19937& rng) const;

  // Load weights from a flat binary written by reference/export_weights.py. The header
  // carries the config, so this also populates cfg. Aborts on a bad magic/version or a
  // size mismatch.
  static GPT load(const std::string& path);
};

}  // namespace moo
