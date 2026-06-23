// MooGPT entry point. Subcommands grow as phases land.
//   demo      Phase 0: hand-checked matmul
//   check     Phase 2: load weights + input, run forward, dump logits (verified vs ref)
//   generate  Phase 2: autoregressive sampling from loaded weights
//   train     Phase 4+: not yet implemented

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "dataloader.h"
#include "loss.h"
#include "model.h"
#include "ops.h"
#include "optimizer.h"
#include "tensor.h"
#include "tokenizer.h"

namespace {

int usage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " <command> [args]\n"
            << "commands:\n"
            << "  demo                                   Phase 0 matmul demo\n"
            << "  check <weights> <input> <out_logits>   forward pass; dump logits\n"
            << "  grads <weights> <input> <targets> <out>  forward+loss+backward; dump grads\n"
            << "  generate <weights> <n> [temp] [top_k]  sample n tokens from id 0\n"
            << "  train <data.txt> [flags]               train a model (see --overfit,\n"
            << "                                         --steps/--batch/--block/--lr/...)\n";
  return 2;
}

int demo() {
  using namespace moo;
  Tensor a({2, 3});
  a.set({1, 2, 3, 4, 5, 6});
  Tensor b({3, 2});
  b.set({7, 8, 9, 10, 11, 12});
  std::cout << "A = " << a << "\n";
  std::cout << "B = " << b << "\n";
  std::cout << "A @ B = " << matmul(a, b) << "\n";
  return 0;
}

// input file format: int32 T, then T int32 token ids.
std::vector<int> read_input_ids(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "cannot open input file: " << path << "\n";
    std::exit(1);
  }
  int32_t T = 0;
  f.read(reinterpret_cast<char*>(&T), sizeof(T));
  std::vector<int> ids(T);
  for (int i = 0; i < T; ++i) {
    int32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), sizeof(v));
    ids[i] = v;
  }
  if (!f) {
    std::cerr << "error reading input ids\n";
    std::exit(1);
  }
  return ids;
}

// logits file format: int32 T, int32 vocab, then T*vocab float32 (row-major).
void write_logits(const std::string& path, const moo::Tensor& logits) {
  std::ofstream f(path, std::ios::binary);
  const int32_t T = logits.dim(0);
  const int32_t V = logits.dim(1);
  f.write(reinterpret_cast<const char*>(&T), sizeof(T));
  f.write(reinterpret_cast<const char*>(&V), sizeof(V));
  f.write(reinterpret_cast<const char*>(logits.data()),
          static_cast<std::streamsize>(logits.size() * sizeof(float)));
}

int check(int argc, char** argv) {
  if (argc < 5) return usage(argv[0]);
  moo::GPT model = moo::GPT::load(argv[2]);
  std::vector<int> ids = read_input_ids(argv[3]);
  moo::Tensor logits = model.forward(ids);
  write_logits(argv[4], logits);
  std::cout << "check: forward ok, wrote logits (" << logits.dim(0) << ", "
            << logits.dim(1) << ") to " << argv[4] << "\n";
  return 0;
}

// grads file format: float32 loss, then every parameter gradient (float32) concatenated
// in the model's canonical order — same order as the weight file.
int grads(int argc, char** argv) {
  if (argc < 6) return usage(argv[0]);
  moo::GPT model = moo::GPT::load(argv[2]);
  std::vector<int> ids = read_input_ids(argv[3]);
  std::vector<int> targets = read_input_ids(argv[4]);

  moo::Tensor logits = model.forward(ids);
  moo::Tensor dlogits;
  float loss = moo::cross_entropy(logits, targets, dlogits);
  model.zero_grad();
  model.backward(dlogits);

  std::ofstream f(argv[5], std::ios::binary);
  f.write(reinterpret_cast<const char*>(&loss), sizeof(loss));
  for (const moo::Tensor* g : model.gradients()) {
    f.write(reinterpret_cast<const char*>(g->data()),
            static_cast<std::streamsize>(g->size() * sizeof(float)));
  }
  std::cout << "grads: loss=" << loss << ", wrote gradients to " << argv[5] << "\n";
  return 0;
}

// --- Phase 4: training ----------------------------------------------------------------

struct TrainArgs {
  std::string data_path;
  std::string out_path = "build/ckpt.bin";
  int steps = 5000;
  int batch_size = 16;
  int block_size = 64;
  int n_layer = 4;
  int n_head = 4;
  int n_embd = 128;
  float lr = 3e-4f;
  float weight_decay = 0.1f;
  unsigned seed = 1337;
  bool overfit = false;  // train on a single fixed batch (the wiring sanity check)
};

// Minimal flag parser: first non-flag positional is the data path; "--key value" sets a
// field; "--overfit" is a boolean. Unknown flags abort loudly.
bool parse_train_args(int argc, char** argv, TrainArgs& a) {
  for (int i = 2; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (s == "--overfit") {
      a.overfit = true;
    } else if (s == "--steps") {
      a.steps = std::atoi(next("--steps").c_str());
    } else if (s == "--batch") {
      a.batch_size = std::atoi(next("--batch").c_str());
    } else if (s == "--block") {
      a.block_size = std::atoi(next("--block").c_str());
    } else if (s == "--layer") {
      a.n_layer = std::atoi(next("--layer").c_str());
    } else if (s == "--head") {
      a.n_head = std::atoi(next("--head").c_str());
    } else if (s == "--embd") {
      a.n_embd = std::atoi(next("--embd").c_str());
    } else if (s == "--lr") {
      a.lr = static_cast<float>(std::atof(next("--lr").c_str()));
    } else if (s == "--wd") {
      a.weight_decay = static_cast<float>(std::atof(next("--wd").c_str()));
    } else if (s == "--seed") {
      a.seed = static_cast<unsigned>(std::strtoul(next("--seed").c_str(), nullptr, 10));
    } else if (s == "--out") {
      a.out_path = next("--out");
    } else if (!s.empty() && s[0] == '-') {
      std::cerr << "unknown flag: " << s << "\n";
      return false;
    } else if (a.data_path.empty()) {
      a.data_path = s;
    } else {
      std::cerr << "unexpected argument: " << s << "\n";
      return false;
    }
  }
  return !a.data_path.empty();
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "cannot open data file: " << path << "\n";
    std::exit(1);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Cosine learning-rate schedule with linear warmup. Ramps 0 -> lr over `warmup` steps,
// then decays lr -> lr*0.1 following half a cosine over the remaining steps.
float lr_schedule(int step, int total, int warmup, float lr) {
  if (step < warmup) return lr * static_cast<float>(step + 1) / static_cast<float>(warmup);
  const float progress =
      static_cast<float>(step - warmup) / static_cast<float>(std::max(1, total - warmup));
  const float coeff = 0.5f * (1.0f + std::cos(3.14159265358979f * progress));
  const float min_lr = lr * 0.1f;
  return min_lr + coeff * (lr - min_lr);
}

// One forward+loss+backward over a whole batch. The model runs one sequence at a time
// (batch=1), so we loop the B rows and accumulate gradients; scaling each row's dlogits
// by 1/B makes the accumulated gradient the gradient of the mean loss over the batch.
// Caller must zero_grad() before this. Returns the mean per-sequence loss.
float train_batch(moo::GPT& model, const moo::Batch& batch) {
  const int B = batch.batch_size;
  const int T = batch.block_size;
  double total = 0.0;
  for (int b = 0; b < B; ++b) {
    std::vector<int> xb(T), yb(T);
    for (int t = 0; t < T; ++t) {
      xb[t] = batch.xat(b, t);
      yb[t] = batch.yat(b, t);
    }
    moo::Tensor logits = model.forward(xb);
    moo::Tensor dlogits;
    total += moo::cross_entropy(logits, yb, dlogits);
    for (std::size_t i = 0; i < dlogits.size(); ++i) dlogits[i] /= static_cast<float>(B);
    model.backward(dlogits);
  }
  return static_cast<float>(total / B);
}

// Sample a short continuation and decode it, so we can eyeball whether text is word-like.
void sample_and_print(moo::GPT& model, const moo::Tokenizer& tok, std::mt19937& rng) {
  const std::vector<char>& chars = tok.chars();
  // Seed from a space if present, else the first vocab char.
  int seed_id = 0;
  for (std::size_t i = 0; i < chars.size(); ++i) {
    if (chars[i] == ' ') { seed_id = static_cast<int>(i); break; }
  }
  std::vector<int> out = model.generate({seed_id}, 200, 0.8f, 40, rng);
  std::cout << "  sample: " << tok.decode(out) << "\n";
}

int train(int argc, char** argv) {
  TrainArgs args;
  if (!parse_train_args(argc, argv, args)) return usage(argv[0]);

  const std::string text = read_file(args.data_path);
  moo::Tokenizer tok = moo::Tokenizer::from_text(text);
  std::vector<int> ids = tok.encode(text);
  std::cout << "corpus: " << text.size() << " chars, vocab=" << tok.vocab_size()
            << ", tokens=" << ids.size() << "\n";
  moo::DataLoader loader(std::move(ids), 0.1f);

  moo::GPTConfig cfg;
  cfg.n_layer = args.n_layer;
  cfg.n_head = args.n_head;
  cfg.n_embd = args.n_embd;
  cfg.block_size = args.block_size;
  cfg.vocab_size = tok.vocab_size();

  std::mt19937 rng(args.seed);
  moo::GPT model = moo::GPT::random_init(cfg, rng);

  moo::AdamWConfig acfg;
  acfg.lr = args.lr;
  // Weight decay fights the overfit sanity check (loss must reach ~0), so disable it there.
  acfg.weight_decay = args.overfit ? 0.0f : args.weight_decay;
  moo::AdamW opt(model.parameters(), acfg);
  std::vector<moo::Tensor*> grads = model.gradients();  // stable addresses, captured once

  if (args.overfit)
    std::cout << "mode: overfit a single fixed batch (loss should approach 0)\n";
  std::cout << "config: layers=" << cfg.n_layer << " heads=" << cfg.n_head
            << " embd=" << cfg.n_embd << " block=" << cfg.block_size
            << " batch=" << args.batch_size << " lr=" << args.lr << "\n";

  const int warmup = std::min(100, std::max(1, args.steps / 20));
  moo::Batch fixed;
  if (args.overfit)
    fixed = loader.get_batch(moo::DataLoader::Split::Train, args.batch_size,
                             cfg.block_size, rng);

  for (int step = 0; step < args.steps; ++step) {
    if (!args.overfit) opt.set_lr(lr_schedule(step, args.steps, warmup, args.lr));

    moo::Batch batch =
        args.overfit ? fixed
                     : loader.get_batch(moo::DataLoader::Split::Train, args.batch_size,
                                        cfg.block_size, rng);
    model.zero_grad();
    const float loss = train_batch(model, batch);
    opt.step(grads);

    if (step % 100 == 0 || step == args.steps - 1) {
      // Flush so progress shows live even when stdout is redirected to a file.
      std::cout << "step " << step << "  loss " << loss << "  lr " << opt.lr()
                << std::endl;
    }
    if (step > 0 && step % 1000 == 0) sample_and_print(model, tok, rng);
  }

  std::cout << "final sample:\n";
  sample_and_print(model, tok, rng);

  model.save(args.out_path);
  std::cout << "saved checkpoint to " << args.out_path << "\n";
  return 0;
}

int generate(int argc, char** argv) {
  if (argc < 4) return usage(argv[0]);
  moo::GPT model = moo::GPT::load(argv[2]);
  const int n = std::atoi(argv[3]);
  const float temp = argc > 4 ? static_cast<float>(std::atof(argv[4])) : 1.0f;
  const int top_k = argc > 5 ? std::atoi(argv[5]) : 0;

  std::mt19937 rng(1234);
  std::vector<int> out = model.generate({0}, n, temp, top_k, rng);
  std::cout << "generated " << out.size() << " token ids:\n";
  for (std::size_t i = 0; i < out.size(); ++i) {
    std::cout << out[i] << (i + 1 < out.size() ? " " : "\n");
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage(argv[0]);
  const char* cmd = argv[1];

  if (std::strcmp(cmd, "demo") == 0) return demo();
  if (std::strcmp(cmd, "check") == 0) return check(argc, argv);
  if (std::strcmp(cmd, "grads") == 0) return grads(argc, argv);
  if (std::strcmp(cmd, "generate") == 0) return generate(argc, argv);
  if (std::strcmp(cmd, "train") == 0) return train(argc, argv);
  return usage(argv[0]);
}
