#include "math_ops.hpp"
#include <cstdlib>
#include <iostream>

static int failures = 0;

#define EXPECT_EQ(a, b) do { if ((a) != (b)) { std::cerr << "fail " << #a << "\n"; ++failures; } } while(0)

int main() {
  EXPECT_EQ(add(2, 3), 5);
  EXPECT_EQ(multiply(4, 5), 20);
  EXPECT_EQ(summarize(1, 2), std::string("sum=3"));
  return failures == 0 ? 0 : 1;
}
