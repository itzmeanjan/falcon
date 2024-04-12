#pragma once
#include "ff.hpp"
#include <cassert>

// Test functional correctness of Falcon PQC suite implementation
namespace test_falcon {

// Test functional correctness of Falcon prime field operations, by running
// through multiple rounds ( see template parameter ) of execution of field
// arithmetic, on randomly sampled field elements
template<const size_t rounds = 1024ul>
static void
test_field_ops()
{
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

    assert(c == e);

    // multiplication, division, inversion
    const auto f = a * b;
    const auto g = f / b;

    if (b == ff::ff_t::zero()) {
      assert(g == ff::ff_t::zero());
    } else {
      assert(g == a);
    }

    const auto h = a.inv();
    const auto k = h * a;

    if (a == ff::ff_t::zero()) {
      assert(k == ff::ff_t::zero());
    } else {
      assert(k == ff::ff_t::one());
    }

    // exponentiation, multiplication
    const size_t exp = dis(gen);
    const auto l = a ^ exp;

    auto res = ff::ff_t::one();
    for (size_t j = 0; j < exp; j++) {
      res = res * a;
    }

    assert(res == l);
  }
}

}
