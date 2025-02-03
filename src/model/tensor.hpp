#pragma once

#include <cstddef>
#include <vector>

struct Shape{
    std::vector<size_t> dims;

    size_t total_size() const{
        size_t size = 1;
        for (auto dim : dims){
            size *= dim;
        }
       return size;
    }
};

class Tensor{
private:
    Shape shape_;
    std::vector<float> data_;

    // Helper to convert multi-dimensional index to flat index
    size_t get_flat_index(const std::vector<size_t>& indices) const;

public:
    // Constructors
    explicit Tensor(const Shape& shape);
    Tensor(const Shape& shape, float fill_value);

    // Basic accessors
    float* data() {return data_.data();}
    const float* data() const {return data_.data();}
    const Shape& shape() const {return shape_;}
    size_t total_size() const {return shape_.total_size();}

   // Element Access
   float& operator()(const std::vector<size_t>& indices);
   const float& operator()(const std::vector<size_t>& indices) const;

   // Simple opeartions
   void fill(float value);

   // Arithmetic Operations
   Tensor operator+(const Tensor& other) const;
   Tensor operator*(const Tensor& other) const; // element-wise multiplication
   Tensor operator*(float scalar) const; // scalar multiplication

   // Matrix Multiplication
   Tensor matmul(const Tensor& other) const;

   // Utilities we'll need for the transformer
   Tensor reshape(const Shape& new_shape) const;
   Tensor transpose(const std::vector<size_t>& dims) const;

   // Common operations needed for attention and normalization
   Tensor softmax(int dim = -1) const;
   float rms() const; // Root Mean Square - needed for RMSNorm
};
