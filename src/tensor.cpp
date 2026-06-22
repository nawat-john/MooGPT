#include "tensor.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>

namespace moo {

namespace {

std::size_t product(const std::vector<int>& dims) {
  std::size_t n = 1;
  for (int d : dims) n *= static_cast<std::size_t>(d);
  return n;
}

}  // namespace

Tensor::Tensor(std::vector<int> shape) : shape_(std::move(shape)) {
  compute_strides();
  data_.assign(product(shape_), 0.0f);
}

Tensor::Tensor(std::initializer_list<int> shape) : Tensor(std::vector<int>(shape)) {}

void Tensor::compute_strides() {
  strides_.assign(shape_.size(), 0);
  std::size_t acc = 1;
  // Row-major: the last axis is contiguous (stride 1), each earlier axis strides
  // by the product of all the axes after it.
  for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
    strides_[i] = static_cast<int>(acc);
    acc *= static_cast<std::size_t>(shape_[i]);
  }
}

std::size_t Tensor::flat_index(std::initializer_list<int> index) const {
  if (static_cast<int>(index.size()) != ndim()) {
    std::cerr << "Tensor::flat_index: index rank " << index.size()
              << " != tensor rank " << ndim() << "\n";
    std::abort();
  }
  std::size_t offset = 0;
  int axis = 0;
  for (int idx : index) {
    offset += static_cast<std::size_t>(idx) * static_cast<std::size_t>(strides_[axis]);
    ++axis;
  }
  return offset;
}

float& Tensor::at(std::initializer_list<int> index) { return data_[flat_index(index)]; }
float Tensor::at(std::initializer_list<int> index) const { return data_[flat_index(index)]; }

float& Tensor::at(int i, int j) { return data_[flat_index({i, j})]; }
float Tensor::at(int i, int j) const { return data_[flat_index({i, j})]; }

void Tensor::fill(float value) {
  for (float& x : data_) x = value;
}

void Tensor::set(std::initializer_list<float> values) {
  if (values.size() != data_.size()) {
    std::cerr << "Tensor::set: got " << values.size() << " values, expected "
              << data_.size() << "\n";
    std::abort();
  }
  std::size_t i = 0;
  for (float v : values) data_[i++] = v;
}

bool Tensor::allclose(const Tensor& other, float atol, float rtol) const {
  if (!same_shape(other)) return false;
  for (std::size_t i = 0; i < data_.size(); ++i) {
    float a = data_[i], b = other.data_[i];
    if (std::fabs(a - b) > atol + rtol * std::fabs(b)) return false;
  }
  return true;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
  os << "Tensor(shape=[";
  for (std::size_t i = 0; i < t.shape().size(); ++i) {
    os << t.shape()[i] << (i + 1 < t.shape().size() ? "," : "");
  }
  os << "], data=[";
  const std::size_t kMax = 16;
  for (std::size_t i = 0; i < t.size() && i < kMax; ++i) {
    os << t[i] << (i + 1 < t.size() && i + 1 < kMax ? ", " : "");
  }
  if (t.size() > kMax) os << ", ...";
  os << "])";
  return os;
}

}  // namespace moo
