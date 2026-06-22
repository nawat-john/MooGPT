// MooGPT entry point. Subcommands grow as phases land; for Phase 0 they are stubs that
// document the intended CLI surface: `train`, `generate`, `check`.

#include <cstring>
#include <iostream>

#include "ops.h"
#include "tensor.h"

namespace {

int usage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " <command>\n"
            << "commands:\n"
            << "  train     train the model (Phase 4+)\n"
            << "  generate  sample from the model (Phase 2+)\n"
            << "  check     run forward/grad checks vs the reference (Phase 2+)\n"
            << "  demo      Phase 0: show a hand-checked matmul\n";
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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage(argv[0]);
  const char* cmd = argv[1];

  if (std::strcmp(cmd, "demo") == 0) return demo();
  if (std::strcmp(cmd, "train") == 0 || std::strcmp(cmd, "generate") == 0 ||
      std::strcmp(cmd, "check") == 0) {
    std::cerr << "'" << cmd << "' is not implemented yet (see PROJECT_PLAN.md roadmap).\n";
    return 1;
  }
  return usage(argv[0]);
}
