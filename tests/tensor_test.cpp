#include "../src/model/tensor.hpp"
#include <gtest/gtest.h>

// Test tensor construction and basic properties
TEST(TensorTest, ConstructionAndBasicProperties) {
  Shape shape{{2, 3}};
  Tensor t(shape);

  EXPECT_EQ(t.total_size(), 6);
  EXPECT_EQ(t.shape().dims, std::vector<size_t>({2, 3}));
}

// Test filling tensor with values
TEST(TensorTest, FillOperation) {
  Shape shape{{2, 2}};
  Tensor t(shape, 5.0f);

  EXPECT_EQ(t({0, 0}), 5.0f);
  EXPECT_EQ(t({0, 1}), 5.0f);
  EXPECT_EQ(t({1, 0}), 5.0f);
  EXPECT_EQ(t({1, 1}), 5.0f);
}

// Test element access and modification
TEST(TensorTest, ElementAccess) {
  Shape shape{{2, 2}};
  Tensor t(shape);

  t({0, 0}) = 1.0f;
  t({0, 1}) = 2.0f;
  t({1, 0}) = 3.0f;
  t({1, 1}) = 4.0f;

  EXPECT_EQ(t({0, 0}), 1.0f);
  EXPECT_EQ(t({0, 1}), 2.0f);
  EXPECT_EQ(t({1, 0}), 3.0f);
  EXPECT_EQ(t({1, 1}), 4.0f);
}

// Test matrix multiplication
TEST(TensorTest, MatrixMultiplication) {
  Shape shape1{{2, 3}};
  Shape shape2{{3, 2}};

  Tensor t1(shape1);
  Tensor t2(shape2);

  // Set values for first matrix
  t1({0, 0}) = 1.0f;
  t1({0, 1}) = 2.0f;
  t1({0, 2}) = 3.0f;
  t1({1, 0}) = 4.0f;
  t1({1, 1}) = 5.0f;
  t1({1, 2}) = 6.0f;

  // Set values for second matrix
  t2({0, 0}) = 7.0f;
  t2({0, 1}) = 8.0f;
  t2({1, 0}) = 9.0f;
  t2({1, 1}) = 10.0f;
  t2({2, 0}) = 11.0f;
  t2({2, 1}) = 12.0f;

  Tensor result = t1.matmul(t2);

  // Expected results:
  // [1*7 + 2*9 + 3*11, 1*8 + 2*10 + 3*12]
  // [4*7 + 5*9 + 6*11, 4*8 + 5*10 + 6*12]
  EXPECT_EQ(result({0, 0}), 58.0f);
  EXPECT_EQ(result({0, 1}), 64.0f);
  EXPECT_EQ(result({1, 0}), 139.0f);
  EXPECT_EQ(result({1, 1}), 154.0f);
}

// Test invalid operations
TEST(TensorTest, InvalidOperations) {
  Shape shape1{{2, 2}};
  Shape shape2{{2, 3}};

  Tensor t1(shape1);
  Tensor t2(shape2);

  // Test invalid addition
  EXPECT_THROW(t1 + t2, std::invalid_argument);

  // Test invalid matrix multiplication
  Shape shape3{{3, 3}};
  Tensor t3(shape3);
  EXPECT_THROW(t1.matmul(t3), std::runtime_error);

  // Test invalid indices
  EXPECT_THROW(t1({0, 0, 0}), std::invalid_argument); // too many indices
  EXPECT_THROW(t1({2, 0}), std::out_of_range);        // index out of bounds
}

// Test reshape operation
TEST(TensorTest, ReshapeOperation) {
  Shape original_shape{{2, 3}};
  Tensor t(original_shape);

  // Fill with sequential values
  float val = 0.0f;
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      t({i, j}) = val++;
    }
  }

  Shape new_shape{{3, 2}};
  Tensor reshaped = t.reshape(new_shape);

  EXPECT_EQ(reshaped.total_size(), 6);
  EXPECT_EQ(reshaped.shape().dims, std::vector<size_t>({3, 2}));

  // Data should be preserved in row-major order
  val = 0.0f;
  for (size_t i = 0; i < 6; i++) {
    EXPECT_EQ(reshaped.data()[i], val++);
  }
}

// Test scalar multiplication
TEST(TensorTest, ScalarMultiplication) {
  Shape shape{{2, 2}};
  Tensor t(shape, 2.0f);

  Tensor result = t * 3.0f;

  EXPECT_EQ(result({0, 0}), 6.0f);
  EXPECT_EQ(result({0, 1}), 6.0f);
  EXPECT_EQ(result({1, 0}), 6.0f);
  EXPECT_EQ(result({1, 1}), 6.0f);
}

// Test element-wise multiplication
TEST(TensorTest, ElementWiseMultiplication) {
  Shape shape{{2, 2}};
  Tensor t1(shape, 2.0f);
  Tensor t2(shape, 3.0f);

  Tensor result = t1 * t2;

  EXPECT_EQ(result({0, 0}), 6.0f);
  EXPECT_EQ(result({0, 1}), 6.0f);
  EXPECT_EQ(result({1, 0}), 6.0f);
  EXPECT_EQ(result({1, 1}), 6.0f);
}

// Test addition
TEST(TensorTest, Addition) {
  Shape shape{{2, 2}};
  Tensor t1(shape, 1.0f);
  Tensor t2(shape, 2.0f);

  Tensor result = t1 + t2;

  EXPECT_EQ(result({0, 0}), 3.0f);
  EXPECT_EQ(result({0, 1}), 3.0f);
  EXPECT_EQ(result({1, 0}), 3.0f);
  EXPECT_EQ(result({1, 1}), 3.0f);
}

// Test transpose
TEST(TensorTest, Transpose) {
  Shape shape{{2, 3}};
  Tensor t(shape);

  // Fill with sequential values
  float val = 0.0f;
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      t({i, j}) = val++;
    }
  }

  Tensor transposed = t.transpose({1, 0}); // Transpose dimensions

  EXPECT_EQ(transposed.shape().dims, std::vector<size_t>({3, 2}));

  // Check if values are correctly transposed
  EXPECT_EQ(transposed({0, 0}), t({0, 0}));
  EXPECT_EQ(transposed({0, 1}), t({1, 0}));
  EXPECT_EQ(transposed({1, 0}), t({0, 1}));
  EXPECT_EQ(transposed({1, 1}), t({1, 1}));
  EXPECT_EQ(transposed({2, 0}), t({0, 2}));
  EXPECT_EQ(transposed({2, 1}), t({1, 2}));
}

TEST(TensorTest, Softmax) {
  Shape shape{{2, 3}};
  Tensor t(shape);
  t({0, 0}) = 1.0f;
  t({0, 1}) = 2.0f;
  t({0, 2}) = 3.0f; // Row 1: [1, 2, 3]
  t({1, 0}) = 0.0f;
  t({1, 1}) = 0.0f;
  t({1, 2}) = 1.0f; // Row 2: [0, 0, 1]

  Tensor result = t.softmax(); // Default dim = -1

  // Expected values for row 1; softmax([1,2,3])
  float sum1 = std::exp(1.0f) + std::exp(2.0f) + std::exp(3.0f);
  EXPECT_NEAR(result({0, 0}), std::exp(1.0f) / sum1, 1e-5); // ~0.090
  EXPECT_NEAR(result({0, 1}), std::exp(2.0f) / sum1, 1e-5); // ~0.245
  EXPECT_NEAR(result({0, 2}), std::exp(3.0f) / sum1, 1e-5); // ~0.665

  // Expected values for row 2: softmax([0, 0, 1])
  float sum2 = std::exp(0.0f) + std::exp(0.0f) + std::exp(1.0f);
  EXPECT_NEAR(result({1, 0}), std::exp(0.0f) / sum2, 1e-5); // ~0.211
  EXPECT_NEAR(result({1, 1}), std::exp(0.0f) / sum2, 1e-5); // ~0.211
  EXPECT_NEAR(result({1, 2}), std::exp(1.0f) / sum2, 1e-5); // ~0.576
                                                            //
  // Verify each row sums to 1
  float row1_sum = result({0, 0}) + result({0, 1}) + result({0, 2});
  float row2_sum = result({1, 0}) + result({1, 1}) + result({1, 2});
  EXPECT_NEAR(row1_sum, 1.0f, 1e-5);
  EXPECT_NEAR(row2_sum, 1.0f, 1e-5);

  // Test invalid dims
  Shape shape1d{{3}};
  Tensor t1d(shape1d);
  EXPECT_THROW(t1d.softmax(), std::invalid_argument);
}

TEST(TensorTest, Rms) {
  Shape shape1d{{3}};
  Tensor t1d(shape1d);
  t1d({0}) = 1.0f;
  t1d({1}) = 2.0f;
  t1d({2}) = 3.0f; // [1, 2, 3]
  Tensor rms1d = t1d.rms();
  EXPECT_EQ(rms1d.shape().dims, std::vector<size_t>({1}));
  EXPECT_NEAR(rms1d({0}),
              std::sqrt((1.0f * 1.0f + 2.0f * 2.0f + 3.0f * 3.0f) / 3.0f),
              1e-5); // ~2.16025

  // Test 2D tensor
  Shape shape2d{{2, 3}};
  Tensor t2d(shape2d);
  t2d({0, 0}) = 1.0f;
  t2d({0, 1}) = 2.0f;
  t2d({0, 2}) = 3.0f; // Row 1: [1, 2, 3]
  t2d({1, 0}) = 0.0f;
  t2d({1, 1}) = 4.0f;
  t2d({1, 2}) = 0.0f; // Row 2: [0, 4, 0]

  Tensor rms2d = t2d.rms(); // Default dim = -1 (last dim)

  EXPECT_EQ(rms2d.shape().dims, std::vector<size_t>({2}));
  // Row 1: sqrt((1^2 + 2^2 + 3^2) / 3) = sqrt(14/3) ≈ 2.16025
  EXPECT_NEAR(rms2d({0}),
              std::sqrt((1.0f * 1.0f + 2.0f * 2.0f + 3.0f * 3.0f) / 3.0f),
              1e-5);
  // Row 2: sqrt((0^2 + 4^2 + 0^2) / 3) = sqrt(16/3) ≈ 2.30940
  EXPECT_NEAR(rms2d({1}),
              std::sqrt((0.0f * 0.0f + 4.0f * 4.0f + 0.0f * 0.0f) / 3.0f),
              1e-5);

  // Test zeros
  Tensor tzero(Shape{{2, 3}});
  tzero.fill(0.0f);
  Tensor rms_zero = tzero.rms();
  EXPECT_NEAR(rms_zero({0}), 0.0f, 1e-5);
  EXPECT_NEAR(rms_zero({1}), 0.0f, 1e-5);

  // Test invalid dims (optional, since only dim=1 is supported for 2D)
  EXPECT_THROW(t2d.rms(0), std::invalid_argument);
}
