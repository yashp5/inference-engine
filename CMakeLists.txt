cmake_minimum_required(VERSION 3.15)
project(inference_engine)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

# Add the tensor library
add_library(tensor
    src/model/tensor.cpp
)

# Main executable
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    fmt::fmt
    tensor
)

# Tests
enable_testing()
add_executable(tensor_tests
    tests/tensor_test.cpp
)
target_link_libraries(tensor_tests
    PRIVATE
    tensor
    GTest::GTest
    GTest::Main
)
