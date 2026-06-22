#include "loss.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace moo {

float cross_entropy(const Tensor& logits, const std::vector<int>& targets,
                    Tensor& dlogits) {
  const int T = logits.dim(0);
  const int V = logits.dim(1);
  if (static_cast<int>(targets.size()) != T) {
    std::cerr << "cross_entropy: targets length " << targets.size()
              << " != T " << T << "\n";
    std::abort();
  }

  dlogits = Tensor({T, V});
  double total = 0.0;
  const float invT = 1.0f / static_cast<float>(T);

  for (int t = 0; t < T; ++t) {
    const float* z = logits.data() + t * V;
    float* dz = dlogits.data() + t * V;
    const int y = targets[t];
    if (y < 0 || y >= V) {
      std::cerr << "cross_entropy: target out of range: " << y << "\n";
      std::abort();
    }

    // Stable softmax.
    float m = z[0];
    for (int j = 1; j < V; ++j) m = z[j] > m ? z[j] : m;
    float sum = 0.0f;
    for (int j = 0; j < V; ++j) sum += std::exp(z[j] - m);
    const float logsum = m + std::log(sum);

    // loss_t = -(z[y] - logsumexp) = logsumexp - z[y]
    total += static_cast<double>(logsum - z[y]);

    // dlogits = (softmax - onehot) / T
    for (int j = 0; j < V; ++j) {
      const float p = std::exp(z[j] - logsum);
      dz[j] = (p - (j == y ? 1.0f : 0.0f)) * invT;
    }
  }

  return static_cast<float>(total) * invT;
}

}  // namespace moo
