# LLM Inference Engine

This project implements a high-performance inference engine for Large Language Models (LLMs), designed to run efficiently on consumer hardware. The engine is built from scratch in C++ with CUDA support for GPU acceleration, focusing on optimized single-batch inference.

## Features

- Lightweight C++ implementation with minimal external dependencies
- Support for both CPU and GPU (CUDA) inference
- FP32 and FP16 precision support
- Memory-efficient tensor operations and attention mechanisms
- KV cache management for optimized token generation
- YALM file format support for model weights

## Requirements

- C++17 compatible compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.15+
- Conan 2.x package manager
- CUDA Toolkit 11.0+ (for GPU acceleration)
- CPU with AVX2 and F16C instruction set support (for FP16 on CPU)

## Building the Project

```bash
# Create and enter build directory
mkdir -p build
cd build

# Install dependencies with Conan 2.x
conan install .. --output-folder=. --build=missing

# Configure with CMake
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build .
```
