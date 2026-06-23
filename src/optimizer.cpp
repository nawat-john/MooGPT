#include "optimizer.h"

#include <cmath>
#include <utility>

namespace moo {

AdamW::AdamW(std::vector<Tensor*> params, const AdamWConfig& cfg)
    : cfg_(cfg), params_(std::move(params)) {
  const std::size_t n = params_.size();
  m_.resize(n);
  v_.resize(n);
  decay_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    m_[i].assign(params_[i]->size(), 0.0f);
    v_[i].assign(params_[i]->size(), 0.0f);
    // GPT-2 convention: decay weight matrices and embeddings, not biases/LayerNorm.
    decay_[i] = params_[i]->ndim() >= 2;
  }
}

void AdamW::step(const std::vector<Tensor*>& grads) {
  ++t_;
  const float b1 = cfg_.beta1;
  const float b2 = cfg_.beta2;
  // Bias-correction divisors: m and v start at 0, so early estimates read low; dividing
  // by (1 - beta^t) rescales them to be unbiased. Both -> 1 as t grows.
  const float bias1 = 1.0f - std::pow(b1, static_cast<float>(t_));
  const float bias2 = 1.0f - std::pow(b2, static_cast<float>(t_));

  for (std::size_t i = 0; i < params_.size(); ++i) {
    float* p = params_[i]->data();
    const float* g = grads[i]->data();
    std::vector<float>& m = m_[i];
    std::vector<float>& v = v_[i];
    const std::size_t n = params_[i]->size();
    const bool decay = decay_[i];

    for (std::size_t j = 0; j < n; ++j) {
      m[j] = b1 * m[j] + (1.0f - b1) * g[j];
      v[j] = b2 * v[j] + (1.0f - b2) * g[j] * g[j];
      const float m_hat = m[j] / bias1;
      const float v_hat = v[j] / bias2;
      // Decoupled weight decay: shrink the parameter itself, separate from the Adam step.
      if (decay) p[j] -= cfg_.lr * cfg_.weight_decay * p[j];
      p[j] -= cfg_.lr * m_hat / (std::sqrt(v_hat) + cfg_.eps);
    }
  }
}

}  // namespace moo
