#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

#include "ops.h"

namespace moo {

namespace {

void die(const std::string& msg) {
  std::cerr << "GPT::load: " << msg << "\n";
  std::abort();
}

// Reads exactly `count` float32 from the stream into a Tensor of the given shape.
Tensor read_tensor(std::ifstream& f, std::vector<int> shape) {
  Tensor t(std::move(shape));
  f.read(reinterpret_cast<char*>(t.data()),
         static_cast<std::streamsize>(t.size() * sizeof(float)));
  if (!f) die("unexpected EOF while reading tensor");
  return t;
}

int32_t read_i32(std::ifstream& f) {
  int32_t v = 0;
  f.read(reinterpret_cast<char*>(&v), sizeof(v));
  if (!f) die("unexpected EOF while reading header");
  return v;
}

}  // namespace

GPT GPT::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) die("cannot open weight file: " + path);

  char magic[4];
  f.read(magic, 4);
  if (!f || std::memcmp(magic, "MGPT", 4) != 0) die("bad magic (expected 'MGPT')");
  const int32_t version = read_i32(f);
  if (version != 1) die("unsupported version " + std::to_string(version));

  GPT m;
  m.cfg.n_layer = read_i32(f);
  m.cfg.n_head = read_i32(f);
  m.cfg.n_embd = read_i32(f);
  m.cfg.block_size = read_i32(f);
  m.cfg.vocab_size = read_i32(f);

  const int C = m.cfg.n_embd;
  const int V = m.cfg.vocab_size;
  const int B = m.cfg.block_size;

  // Canonical tensor order — MUST match reference/export_weights.py exactly.
  m.wte = read_tensor(f, {V, C});
  m.wpe = read_tensor(f, {B, C});

  m.blocks.resize(m.cfg.n_layer);
  for (auto& blk : m.blocks) {
    blk.cfg = m.cfg;
    blk.attn.cfg = m.cfg;
    blk.mlp.cfg = m.cfg;

    blk.ln_1_w = read_tensor(f, {C});
    blk.ln_1_b = read_tensor(f, {C});
    blk.attn.c_attn_w = read_tensor(f, {3 * C, C});
    blk.attn.c_attn_b = read_tensor(f, {3 * C});
    blk.attn.c_proj_w = read_tensor(f, {C, C});
    blk.attn.c_proj_b = read_tensor(f, {C});
    blk.ln_2_w = read_tensor(f, {C});
    blk.ln_2_b = read_tensor(f, {C});
    blk.mlp.c_fc_w = read_tensor(f, {4 * C, C});
    blk.mlp.c_fc_b = read_tensor(f, {4 * C});
    blk.mlp.c_proj_w = read_tensor(f, {C, 4 * C});
    blk.mlp.c_proj_b = read_tensor(f, {C});
  }

  m.ln_f_w = read_tensor(f, {C});
  m.ln_f_b = read_tensor(f, {C});
  m.lm_head_w = read_tensor(f, {V, C});

  // There should be nothing left but EOF.
  f.peek();
  if (!f.eof()) die("trailing bytes in weight file (config/order mismatch?)");
  return m;
}

Tensor GPT::forward(const std::vector<int>& tokens) {
  const int T = static_cast<int>(tokens.size());
  const int C = cfg.n_embd;
  if (T == 0) die("forward: empty token sequence");
  if (T > cfg.block_size) die("forward: sequence longer than block_size");
  tokens_ = tokens;

  // x = token embedding + positional embedding.
  Tensor x({T, C});
  for (int t = 0; t < T; ++t) {
    const int id = tokens[t];
    if (id < 0 || id >= cfg.vocab_size) die("forward: token id out of range");
    const float* tok = wte.data() + static_cast<std::size_t>(id) * C;
    const float* pos = wpe.data() + static_cast<std::size_t>(t) * C;
    float* xr = x.data() + static_cast<std::size_t>(t) * C;
    for (int c = 0; c < C; ++c) xr[c] = tok[c] + pos[c];
  }

  for (Block& blk : blocks) x = blk.forward(x);
  blocks_out_ = x;
  lnf_out_ = layernorm(x, ln_f_w, ln_f_b);

  // Logits: (T, C) @ lm_head_w^T -> (T, vocab). No bias.
  return linear(lnf_out_, lm_head_w, Tensor());
}

void GPT::backward(const Tensor& dlogits) {
  const int T = static_cast<int>(tokens_.size());
  const int C = cfg.n_embd;

  // lm_head (no bias): logits = lnf_out_ @ lm_head_w^T.
  Tensor no_bias;
  Tensor dx = linear_backward(lnf_out_, lm_head_w, dlogits, d_lm_head_w, no_bias);

  // Final LayerNorm.
  dx = layernorm_backward(blocks_out_, ln_f_w, dx, d_ln_f_w, d_ln_f_b);

  // Transformer blocks, in reverse.
  for (int l = static_cast<int>(blocks.size()) - 1; l >= 0; --l) {
    dx = blocks[l].backward(dx);
  }

  // Embeddings: x[t] = wte[token[t]] + wpe[t], so the gradient at row t routes to both
  // the token's embedding row and position t's embedding row (accumulated — a repeated
  // token sums contributions across its positions).
  for (int t = 0; t < T; ++t) {
    const int id = tokens_[t];
    const float* dxr = dx.data() + static_cast<std::size_t>(t) * C;
    float* dwte = d_wte.data() + static_cast<std::size_t>(id) * C;
    float* dwpe = d_wpe.data() + static_cast<std::size_t>(t) * C;
    for (int c = 0; c < C; ++c) {
      dwte[c] += dxr[c];
      dwpe[c] += dxr[c];
    }
  }
}

void GPT::zero_grad() {
  d_wte = Tensor(wte.shape());
  d_wpe = Tensor(wpe.shape());
  d_ln_f_w = Tensor(ln_f_w.shape());
  d_ln_f_b = Tensor(ln_f_b.shape());
  d_lm_head_w = Tensor(lm_head_w.shape());
  for (Block& blk : blocks) blk.zero_grad();
}

std::vector<Tensor*> GPT::parameters() {
  std::vector<Tensor*> p;
  p.push_back(&wte);
  p.push_back(&wpe);
  for (Block& b : blocks) {
    p.push_back(&b.ln_1_w);
    p.push_back(&b.ln_1_b);
    p.push_back(&b.attn.c_attn_w);
    p.push_back(&b.attn.c_attn_b);
    p.push_back(&b.attn.c_proj_w);
    p.push_back(&b.attn.c_proj_b);
    p.push_back(&b.ln_2_w);
    p.push_back(&b.ln_2_b);
    p.push_back(&b.mlp.c_fc_w);
    p.push_back(&b.mlp.c_fc_b);
    p.push_back(&b.mlp.c_proj_w);
    p.push_back(&b.mlp.c_proj_b);
  }
  p.push_back(&ln_f_w);
  p.push_back(&ln_f_b);
  p.push_back(&lm_head_w);
  return p;
}

std::vector<Tensor*> GPT::gradients() {
  std::vector<Tensor*> g;
  g.push_back(&d_wte);
  g.push_back(&d_wpe);
  for (Block& b : blocks) {
    g.push_back(&b.d_ln_1_w);
    g.push_back(&b.d_ln_1_b);
    g.push_back(&b.attn.d_c_attn_w);
    g.push_back(&b.attn.d_c_attn_b);
    g.push_back(&b.attn.d_c_proj_w);
    g.push_back(&b.attn.d_c_proj_b);
    g.push_back(&b.d_ln_2_w);
    g.push_back(&b.d_ln_2_b);
    g.push_back(&b.mlp.d_c_fc_w);
    g.push_back(&b.mlp.d_c_fc_b);
    g.push_back(&b.mlp.d_c_proj_w);
    g.push_back(&b.mlp.d_c_proj_b);
  }
  g.push_back(&d_ln_f_w);
  g.push_back(&d_ln_f_b);
  g.push_back(&d_lm_head_w);
  return g;
}

std::vector<int> GPT::generate(const std::vector<int>& prompt, int max_new_tokens,
                               float temperature, int top_k, std::mt19937& rng) {
  std::vector<int> seq = prompt;
  const int V = cfg.vocab_size;

  for (int step = 0; step < max_new_tokens; ++step) {
    // Crop to the last block_size tokens (the model can't see further back).
    std::vector<int> ctx = seq;
    if (static_cast<int>(ctx.size()) > cfg.block_size) {
      ctx.erase(ctx.begin(), ctx.end() - cfg.block_size);
    }

    Tensor logits = forward(ctx);
    const int T = static_cast<int>(ctx.size());

    // Take the last row's logits — the distribution over the next token.
    std::vector<float> row(V);
    const float* lr = logits.data() + static_cast<std::size_t>(T - 1) * V;
    const float temp = temperature > 0.0f ? temperature : 1.0f;
    for (int i = 0; i < V; ++i) row[i] = lr[i] / temp;

    // Optional top-k: keep the k largest logits, mask the rest to -inf.
    if (top_k > 0 && top_k < V) {
      std::vector<float> sorted = row;
      std::nth_element(sorted.begin(), sorted.begin() + (V - top_k), sorted.end());
      const float kth = sorted[V - top_k];  // k-th largest value
      for (int i = 0; i < V; ++i) {
        if (row[i] < kth) row[i] = -std::numeric_limits<float>::infinity();
      }
    }

    // Softmax -> sample.
    float m = -std::numeric_limits<float>::infinity();
    for (float v : row) m = v > m ? v : m;
    float sum = 0.0f;
    for (int i = 0; i < V; ++i) {
      row[i] = std::exp(row[i] - m);
      sum += row[i];
    }
    for (int i = 0; i < V; ++i) row[i] /= sum;

    std::discrete_distribution<int> dist(row.begin(), row.end());
    seq.push_back(dist(rng));
  }
  return seq;
}

}  // namespace moo
