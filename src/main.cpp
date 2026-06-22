// MooGPT entry point. Subcommands grow as phases land.
//   demo      Phase 0: hand-checked matmul
//   check     Phase 2: load weights + input, run forward, dump logits (verified vs ref)
//   generate  Phase 2: autoregressive sampling from loaded weights
//   train     Phase 4+: not yet implemented

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "loss.h"
#include "model.h"
#include "ops.h"
#include "tensor.h"

namespace {

int usage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " <command> [args]\n"
            << "commands:\n"
            << "  demo                                   Phase 0 matmul demo\n"
            << "  check <weights> <input> <out_logits>   forward pass; dump logits\n"
            << "  grads <weights> <input> <targets> <out>  forward+loss+backward; dump grads\n"
            << "  generate <weights> <n> [temp] [top_k]  sample n tokens from id 0\n"
            << "  train                                  not implemented yet\n";
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
  if (std::strcmp(cmd, "train") == 0) {
    std::cerr << "'train' is not implemented yet (Phase 4).\n";
    return 1;
  }
  return usage(argv[0]);
}
