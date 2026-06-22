#pragma once

namespace moo {

// GPT hyperparameters. Kept tiny by default (Phase 4 char-level sizing). The reference
// exporter writes these into the weight file header so the C++ loader can't silently
// disagree with the architecture the weights were created for.
struct GPTConfig {
  int n_layer = 4;
  int n_head = 4;
  int n_embd = 128;
  int block_size = 128;  // max context length
  int vocab_size = 0;    // set from the tokenizer / weight file

  int head_dim() const { return n_embd / n_head; }
};

}  // namespace moo
