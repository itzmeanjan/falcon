#include "ff.hpp"
#include <gtest/gtest.h>

// Test functional correctness of Falcon prime field operations, by running
// through multiple rounds of execution of field arithmetic, on randomly sampled
// field elements.
TEST(Falcon, ArithmeticOverZq)
{
  constexpr size_t rounds = 1024ul;

  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<size_t> dis{ 0ul, 1ul << 20 };

  for (size_t i = 0; i < rounds; i++) {
    const auto a = ff::ff_t::random();
    const auto b = ff::ff_t::random();

    // addition, subtraction, negation
    const auto c = a - b;
    const auto d = -b;
    const auto e = a + d;

    EXPECT_EQ(c, e);

    // multiplication, division, inversion
    const auto f = a * b;
    const auto g = f / b;

    if (b == ff::ff_t::zero()) {
      EXPECT_EQ(g, ff::ff_t::zero());
    } else {
      EXPECT_EQ(g, a);
    }

    const auto h = a.inv();
    const auto k = h * a;

    if (a == ff::ff_t::zero()) {
      EXPECT_EQ(k, ff::ff_t::zero());
    } else {
      EXPECT_EQ(k, ff::ff_t::one());
    }

    // exponentiation, multiplication
    const size_t exp = dis(gen);
    const auto l = a ^ exp;

    auto res = ff::ff_t::one();
    for (size_t j = 0; j < exp; j++) {
      res = res * a;
    }

    EXPECT_EQ(res, l);
  }
}
