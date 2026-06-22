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
// makes the forward/backward and the weight export trivially explicit. Revisit tying
// once correctness is locked in.
class GPT {
 public:
  GPTConfig cfg;

  // Parameters.
  Tensor wte;        // (vocab, C) token embedding
  Tensor wpe;        // (block_size, C) positional embedding
  std::vector<Block> blocks;
  Tensor ln_f_w, ln_f_b;  // (C,)
  Tensor lm_head_w;       // (vocab, C), no bias

  // Gradients for the model-level parameters (blocks own their own grads).
  Tensor d_wte, d_wpe, d_ln_f_w, d_ln_f_b, d_lm_head_w;

  // Forward pass on a single sequence of token ids (length T <= block_size).
  // Caches intermediates for backward(); returns logits of shape (T, vocab_size).
  Tensor forward(const std::vector<int>& tokens);

  // Backprop from dlogits (T, vocab) through the whole model, accumulating every
  // parameter gradient. Call zero_grad() first.
  void backward(const Tensor& dlogits);

  // Allocate/zero all gradient buffers (model-level and every block's).
  void zero_grad();

  // Parameters and their gradients in one canonical order (parallel vectors). This is
  // the SAME order used by the weight/grad binary format, so it drives both the
  // finite-difference check and the gradient export.
  std::vector<Tensor*> parameters();
  std::vector<Tensor*> gradients();

  // Autoregressive sampling (see generate()): temperature > 0 scales logits;
  // top_k <= 0 disables top-k filtering.
  std::vector<int> generate(const std::vector<int>& prompt, int max_new_tokens,
                            float temperature, int top_k, std::mt19937& rng);

  // Load weights from a flat binary written by reference/export_weights.py.
  static GPT load(const std::string& path);

 private:
  std::vector<int> tokens_;  // cached input ids
  Tensor blocks_out_;        // cached: output of the last block (input to ln_f)
  Tensor lnf_out_;           // cached: ln_f output (input to lm_head)
};

}  // namespace moo
