#pragma once

#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <vector>

namespace moo {

// A dense, row-major, fp32 tensor.
//
// The data lives in a single contiguous std::vector<float>. "Row-major" means the
// last dimension is the fastest-varying one in memory: for a 2-D tensor of shape
// (rows, cols), element (i, j) sits at flat index i*cols + j. Generalizing, the flat
// offset of a full index is sum_k(index[k] * stride[k]), where stride[k] is the
// product of all dimension sizes after k. Keeping one flat buffer (instead of nested
// vectors) is what makes the math cache-friendly and easy to hand off to BLAS/CUDA later.
class Tensor {
 public:
  Tensor() = default;

  // Construct with a shape; the buffer is zero-initialized.
  explicit Tensor(std::vector<int> shape);
  Tensor(std::initializer_list<int> shape);

  // Shape / size info.
  const std::vector<int>& shape() const { return shape_; }
  const std::vector<int>& strides() const { return strides_; }
  int ndim() const { return static_cast<int>(shape_.size()); }
  int dim(int i) const { return shape_[i]; }
  std::size_t size() const { return data_.size(); }  // total element count

  // Raw contiguous buffer access.
  float* data() { return data_.data(); }
  const float* data() const { return data_.data(); }
  std::vector<float>& buffer() { return data_; }
  const std::vector<float>& buffer() const { return data_; }

  // Flat indexing into the contiguous buffer.
  float& operator[](std::size_t i) { return data_[i]; }
  float operator[](std::size_t i) const { return data_[i]; }

  // Multi-dimensional indexing (computes the flat offset via strides).
  float& at(std::initializer_list<int> index);
  float at(std::initializer_list<int> index) const;

  // Convenience 2-D accessors (assert ndim()==2).
  float& at(int i, int j);
  float at(int i, int j) const;

  // Fill helpers.
  void fill(float value);
  void zero() { fill(0.0f); }

  // Copy values from a flat list (must match size()).
  void set(std::initializer_list<float> values);

  // Structural equality of shapes.
  bool same_shape(const Tensor& other) const { return shape_ == other.shape_; }

  // Approximate elementwise equality (for tests).
  bool allclose(const Tensor& other, float atol = 1e-5f, float rtol = 1e-5f) const;

  std::size_t flat_index(std::initializer_list<int> index) const;

 private:
  void compute_strides();

  std::vector<int> shape_;
  std::vector<int> strides_;
  std::vector<float> data_;
};

std::ostream& operator<<(std::ostream& os, const Tensor& t);

}  // namespace moo
