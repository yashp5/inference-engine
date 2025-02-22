# TinyLlama Inference Engine

This project is a C++ implementation of an inference engine for [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B), a lightweight language model based on the Llama architecture. The goal is to build a working inference engine from scratch, without relying on external tensor libraries like Eigen or BLAS. Optimization (e.g., speed, memory efficiency) will be addressed in a later phase. This README provides a roadmap to set up, develop, and run the engine with TinyLlama-1.1B weights.

---

## Project Structure

```
├── CMakeLists.txt                # Build configuration
├── CMakeUserPresets.json         # CMake build presets
├── README.md                     # This file
├── compile_commands.json         # Symlink to build/compile_commands.json
├── conanfile.txt                 # Conan dependencies (minimal)
├── src
│   ├── main.cpp                  # Entry point
│   ├── model
│   │   ├── config.hpp            # Model hyperparameters
│   │   ├── model.cpp             # TinyLlama implementation
│   │   ├── model.hpp             # Model interface
│   │   ├── tensor.cpp            # Custom tensor operations
│   │   ├── tensor.hpp            # Tensor class
│   │   ├── layer_norm.cpp        # Layer normalization
│   │   ├── layer_norm.hpp
│   │   ├── attention.cpp         # Multi-head attention
│   │   ├── attention.hpp
│   │   ├── feed_forward.cpp      # Feed-forward network
│   │   ├── feed_forward.hpp
│   │   ├── transformer_block.cpp # Transformer block
│   │   ├── transformer_block.hpp
│   │   ├── embedding.cpp         # Embedding layer
│   │   ├── embedding.hpp
│   │   ├── weights_loader.cpp    # Load TinyLlama weights
│   │   └── weights_loader.hpp
│   ├── tokenizer
│   │   ├── tokenizer.cpp         # Tokenizer implementation
│   │   └── tokenizer.hpp         # Tokenizer interface
│   └── utils
│       ├── softmax.cpp           # Softmax utility
│       ├── softmax.hpp
│       ├── sampling.cpp          # Sampling (greedy, top-k)
│       └── sampling.hpp
└── tests
    ├── tensor_test.cpp           # Tensor tests
    ├── model_test.cpp            # Model tests
    ├── attention_test.cpp        # Attention tests
    └── tokenizer_test.cpp        # Tokenizer tests
```

---

## Prerequisites

- **C++ Compiler**: C++17 or later (e.g., GCC, Clang).
- **CMake**: Version 3.20 or higher.
- **Conan**: Package manager for dependencies (optional, used only for JSON parsing if needed).
- **Python**: For converting TinyLlama-1.1B weights (optional).
- **TinyLlama-1.1B Weights**: Pretrained weights from [Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B) or equivalent.

---

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd tinyllama-inference
   ```

2. **Install Conan (Optional)**:
   If you need JSON parsing (e.g., for weights or tokenizer vocab):
   ```bash
   pip install conan
   ```

3. **Prepare TinyLlama-1.1B Weights**:
   - Download the TinyLlama-1.1B model from Hugging Face.
   - Convert weights to a C++-readable format (e.g., raw binary files):
     ```python
     from transformers import AutoModelForCausalLM
     model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B")
     for name, param in model.named_parameters():
         param.detach().cpu().numpy().tofile(f"weights/{name}.bin")
     ```
   - Store weights in a `weights/` directory (create it in the project root).

4. **Configure Build**:
   ```bash
   mkdir build
   cd build
   cmake .. -DCMAKE_BUILD_TYPE=Debug
   ```

5. **Build**:
   ```bash
   cmake --build .
   ```

---

## Development Roadmap

Follow these steps to build the inference engine. Focus is on functionality; optimization comes later.

### 1. Project Setup
- Update `CMakeLists.txt` to include all `.cpp` files in `src/`.
- Add minimal dependencies to `conanfile.txt` (e.g., `nlohmann_json` if needed for weights parsing).
- Test the build with an empty `main.cpp`.

### 2. Custom Tensor Operations
- In `tensor.hpp`/`tensor.cpp`, define a basic `Tensor` class:
  - Store data as a flat `std::vector<float>` with rows and columns.
  - Implement matrix multiplication, addition, and transposition manually.
- Test operations in `tensor_test.cpp` with small matrices.

### 3. Model Architecture
- **Config**: Define TinyLlama-1.1B hyperparameters in `config.hpp` (e.g., vocab_size=32000, hidden_size=2048, num_layers=22, num_heads=32).
- **Weights Loader**: In `weights_loader`, read `.bin` files into `Tensor` objects.
- **Embedding**: In `embedding`, map token IDs to vectors using the embedding weights.
- **Attention**: In `attention`, implement multi-head self-attention (Q, K, V projections, scaled dot-product).
- **Feed-Forward**: In `feed_forward`, build the FFN with two linear layers and an activation (e.g., SwiGLU).
- **Layer Norm**: In `layer_norm`, normalize inputs across the hidden dimension.
- **Transformer Block**: In `transformer_block`, combine attention, FFN, and normalization with residuals.
- **Full Model**: In `model`, stack embeddings, transformer blocks, and an output layer (hidden_size → vocab_size).

### 4. Tokenization
- In `tokenizer`, implement a basic tokenizer:
  - Load TinyLlama’s vocabulary (e.g., from `tokenizer.json` or `.model`).
  - Convert text to token IDs and back.
- Test in `tokenizer_test.cpp`.

### 5. Inference Logic
- In `model`, implement the forward pass: tokens → embeddings → transformer blocks → logits.
- In `softmax`, compute probabilities from logits.
- In `sampling`, add greedy sampling (later extend to top-k if desired).
- In `main.cpp`, tie it together: load weights, tokenize input, run inference, decode output.

### 6. Testing
- Write unit tests for each component (tensors, attention, model, tokenizer).
- Validate end-to-end with a small input (e.g., “Hello”) against Python outputs.

### 7. Integration
- Finalize `main.cpp` to accept command-line input and print generated text.
- Ensure weights load correctly and inference runs without crashes.

---

## Running the Engine

Once implemented:
1. Build the project:
   ```bash
   cd build
   cmake --build .
   ```
2. Run with a sample input:
   ```bash
   ./inference "Hello, how are you?"
   ```
3. Expected output: Generated text based on TinyLlama-1.1B.

---

## Notes
- **Weights**: TinyLlama-1.1B has 1.1 billion parameters (~4.4GB in FP32). Ensure your system has enough RAM.
- **Tokenizer**: Use TinyLlama’s provided tokenizer vocab for compatibility.
- **Performance**: This initial version prioritizes correctness over speed. Optimization (e.g., multi-threading, quantization) is a future step.

---

## Troubleshooting
- **Build Errors**: Check `CMakeLists.txt` for missing files or dependencies.
- **Weights Issues**: Verify `.bin` files match TinyLlama’s structure.
- **Output Gibberish**: Debug tokenization or model forward pass.

---

## Future Improvements
- Optimize tensor operations with manual loop unrolling or SIMD.
- Add support for FP16 or INT8 weights.
- Implement advanced sampling (top-k, top-p).
- Port to GPU with CUDA (optional).

---

This README gives you a clear path forward without overwhelming you with code details. Start with the setup, then tackle each section step-by-step. Let me know if you need clarification on any part!