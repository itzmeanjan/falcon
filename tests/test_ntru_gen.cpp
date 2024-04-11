#include "check_ntru_eq.hpp"
#include "prng.hpp"
#include <gtest/gtest.h>

// Test functional correctness of NTRUGen routine, by first generating f, g, F,
// G ∈ Z[x]/(x^N + 1) and then solving NTRU equation ( see eq 3.15 of Falcon
// specification ).
//
// Collects some inspiration from
// https://github.com/tprest/falcon.py/blob/88d01ed/test.py#L88-L94
template<const size_t N>
static void
test_ntru_gen()
{
  auto f = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto g = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto F = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto G = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));

  prng::prng_t prng;
  ntru_gen::ntru_gen<N>(f, g, F, G, prng);
  const bool flg = test_falcon::check_ntru_eq<N>(f, g, F, G);

  std::free(f);
  std::free(g);
  std::free(F);
  std::free(G);

  EXPECT_TRUE(flg);
}

TEST(Falcon, NTRUGen)
{
  test_ntru_gen<ntt::FALCON512_N>();
  test_ntru_gen<ntt::FALCON1024_N>();
}
