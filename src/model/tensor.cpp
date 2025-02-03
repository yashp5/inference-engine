#include "tensor.hpp"
#include <algorithm>
#include <functional>
#include <vector>

Tensor::Tensor(const Shape &shape) : shape_(shape) {
  data_.resize(shape.total_size());
}

Tensor::Tensor(const Shape& shape, float fill_value) : shape_(shape) {
    data_.resize(shape.total_size(), fill_value);
}

void Tensor::fill(float value) {
    data_.resize(shape_.total_size(), value);
    std::fill(data_.begin(), data_.end(), value);
}

size_t Tensor::get_flat_index(const std::vector<size_t> &indices) const {
  if (indices.size() != shape_.dims.size()) {
    throw std::invalid_argument("Invalid number of indices");
  }

  size_t flat_index = 0;
  size_t stride = 1;

  for (int i = indices.size() - 1; i >= 0; i--) {
    if (indices[i] >= shape_.dims[i]) {
      throw std::out_of_range("Index out of range");
    }
    flat_index += indices[i] * stride;
    stride *= shape_.dims[i];
  }

  return flat_index;
}

float &Tensor::operator()(const std::vector<size_t> &indices) {
  return data_[get_flat_index(indices)];
}

const float &Tensor::operator()(const std::vector<size_t> &indices) const {
  return data_[get_flat_index(indices)];
}

Tensor Tensor::operator+(const Tensor &other) const {
  if (shape_.dims != other.shape_.dims) {
    throw std::invalid_argument("Shape mismatch for addition");
  }

  Tensor result(shape_);
  for (size_t i = 0; i < data_.size(); i++) {
    result.data_[i] = data_[i] + other.data_[i];
  }

  return result;
}

Tensor Tensor::operator*(const Tensor &other) const {
  if (shape_.dims != other.shape_.dims) {
    throw std::invalid_argument("Shape mismatch for addition");
  }

  Tensor result(shape_);
  for (size_t i = 0; i <= data_.size(); i++) {
    result.data_[i] = data_[i] * other.data_[i];
  }
  return result;
}

Tensor Tensor::operator*(float scalar) const {
  Tensor result(shape_);
  for (size_t i = 0; i < data_.size(); i++) {
    result.data_[i] = data_[i] * scalar;
  }
  return result;
}

Tensor Tensor::matmul(const Tensor &other) const {
  if (shape_.dims.size() != 2 || other.shape_.dims.size() != 2 ||
      shape_.dims[1] != other.shape_.dims[0]) {
    throw std::runtime_error("Invalid shapes for matrix multiplication");
  }

  Shape result_shape;
  result_shape.dims = {shape_.dims[0], other.shape_.dims[1]};
  Tensor result(result_shape);

  // Simple triple-loop matrix multiplication
  for (size_t i = 0; i < shape_.dims[0]; i++) {
    for (size_t j = 0; j < other.shape_.dims[1]; j++) {
      float sum = 0.0f;
      for (size_t k = 0; k < shape_.dims[1]; k++) {
        sum += (*this)({i, k}) * other({k, j});
      }
      result({i, j}) = sum;
    }
  }

  return result;
}

Tensor Tensor::reshape(const Shape &new_shape) const {
  // Check if the new shape has the same total size
  if (new_shape.total_size() != shape_.total_size()) {
    throw std::invalid_argument("New shape must have the same total size");
  }

  Tensor result(new_shape);
  // Copy the data directly since we're just changing the logical shape
  std::copy(data_.begin(), data_.end(), result.data_.begin());
  return result;
}

Tensor Tensor::transpose(const std::vector<size_t> &dims) const {
  // Check if the permutation is valid
  if (dims.size() != shape_.dims.size()) {
    throw std::invalid_argument(
        "Transpose dimensions must match tensor dimensions");
  }

  // Create new shape based on the permutation
  Shape new_shape;
  new_shape.dims.resize(dims.size());
  for (size_t i = 0; i < dims.size(); i++) {
    new_shape.dims[i] = shape_.dims[dims[i]];
  }

  Tensor result(new_shape);

  // Helper lambda to iterate through all indices
  std::function<void(std::vector<size_t> &, size_t)> iterate_indices;
  iterate_indices = [&](std::vector<size_t> &indices, size_t dim) {
    if (dim == indices.size()) {
      // Permute indices according to dims and copy the value
      std::vector<size_t> permuted_indices(indices.size());
      for (size_t i = 0; i < indices.size(); i++) {
        permuted_indices[i] = indices[dims[i]];
      }
      result(permuted_indices) = (*this)(indices);
      return;
    }

    for (size_t i = 0; i < shape_.dims[dim]; i++) {
      indices[dim] = i;
      iterate_indices(indices, dim + 1);
    }
  };

  std::vector<size_t> indices(shape_.dims.size(), 0);
  iterate_indices(indices, 0);

  return result;
}
