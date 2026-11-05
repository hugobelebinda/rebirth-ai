#include "math_ops.hpp"

int add(int a, int b) { return a + b; }

int multiply(int a, int b) {
  // BUG: should multiply
  return a + b;
}

std::string summarize(int a, int b) {
  return "sum=" + std::to_string(add(a, b));
}
