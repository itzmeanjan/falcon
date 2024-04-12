#include "polynomial.hpp"
#include <cstring>
#include <gtest/gtest.h>

// Ensure functional correctness of (i)NTT implementation, using polynomial
// multiplication and division over Z_q
//
// Test is adapted from
// https://github.com/tprest/falcon.py/blob/88d01ed/test.py#L62-L77
template<const size_t lgn>
static void
test_ntt()
{
  const size_t n = 1ul << lgn;

  auto* poly_a = static_cast<ff::ff_t*>(std::malloc(n * sizeof(ff::ff_t)));
  auto* poly_b = static_cast<ff::ff_t*>(std::malloc(n * sizeof(ff::ff_t)));
  auto* poly_d = static_cast<ff::ff_t*>(std::malloc(n * sizeof(ff::ff_t)));

  auto* ntt_a = static_cast<ff::ff_t*>(std::malloc(n * sizeof(ff::ff_t)));
  auto* ntt_b = static_cast<ff::ff_t*>(std::malloc(n * sizeof(ff::ff_t)));
  auto* ntt_c = static_cast<ff::ff_t*>(std::malloc(n * sizeof(ff::ff_t)));
  auto* ntt_d = static_cast<ff::ff_t*>(std::malloc(n * sizeof(ff::ff_t)));

  bool flg = false;

  while (1) {
    for (size_t i = 0; i < n; i++) {
      poly_a[i] = ff::ff_t::random();
      poly_b[i] = ff::ff_t::random();
    }

    std::memcpy(ntt_a, poly_a, n * sizeof(ff::ff_t));
    std::memcpy(ntt_b, poly_b, n * sizeof(ff::ff_t));

    ntt::ntt<lgn>(ntt_a);
    ntt::ntt<lgn>(ntt_b);

    bool can_invert = true;
    for (size_t i = 0; i < n; i++) {
      can_invert &= ntt_b[i].v != 0;
    }

    if (!can_invert) {
      continue;
    }

    polynomial::mul<lgn>(ntt_a, ntt_b, ntt_c); // c = a * b
    polynomial::div<lgn>(ntt_c, ntt_b, ntt_d); // d = c / b

    std::memcpy(poly_d, ntt_d, n * sizeof(ff::ff_t));
    ntt::intt<lgn>(poly_d);

    flg = false; // just to be safe
    for (size_t i = 0; i < n; i++) {
      flg |= static_cast<bool>(poly_d[i].v ^ poly_a[i].v);
    }

    break;
  }

  std::free(poly_a);
  std::free(poly_b);
  std::free(poly_d);
  std::free(ntt_a);
  std::free(ntt_b);
  std::free(ntt_c);
  std::free(ntt_d);

  EXPECT_FALSE(flg);
}

TEST(Falcon, NumberTheoreticTransform)
{
  test_ntt<ntt::FALCON512_LOG2N>();
  test_ntt<ntt::FALCON1024_LOG2N>();
}
