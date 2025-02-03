// src/main.cpp
#include <fmt/core.h>

int main() {
  // Similar to fmt.Printf in Go
  fmt::print("Hello, {}!\n", "World");

  // Similar to fmt.Sprintf in Go
  std::string message = fmt::format("The answer is {}", 42);
  fmt::print("{}\n", message);

  return 0;
}
