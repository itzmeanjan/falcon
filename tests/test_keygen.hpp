#pragma once
#include "decoding.hpp"
#include "encoding.hpp"
#include "falcon.hpp"
#include "ntru_gen.hpp"
#include "prng.hpp"
#include "test_ntru_gen.hpp"
#include <cassert>
#include <cstdlib>

// Test functional correctness of Falcon PQC suite implementation
namespace test_falcon {

// Test if Falcon Key Generation Algorithm works as expected by doing following
//
// - First generate f, g, F and G
// - Encode f, g and F into secret key
// - Deserialize secret key bytes to get back f, g and F
// - Recompute G using NTRU equation
// - Check if NTRU equation still satisfies or not
// - Also ensure that actual G and recomputed G' matches
template<const size_t N>
void
test_keygen()
{
  constexpr size_t sklen = falcon_utils::compute_skey_len<N>();

  auto f = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto g = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto F = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto G = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto f_ = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto g_ = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto F_ = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto G_ = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto skey = static_cast<uint8_t*>(std::malloc(sklen));
  prng::prng_t rng;

  // Generate f, g, F and G
  ntru_gen::ntru_gen<N>(f, g, F, G, rng);
  // Encode f, g and F
  encoding::encode_skey<N>(f, g, F, skey);
  // Deserialize f, g and F
  decoding::decode_skey<N>(skey, f_, g_, F_);
  // Recompute G using NTRU equation
  falcon::recompute_G<N>(f_, g_, F_, G_);
  // See if NTRU equation can be solved
  const bool flg = check_ntru_eq<N>(f_, g_, F_, G_);

  // Ensure that each coefficient of original G and recomputed G' matches
  bool match = true;
  for (size_t i = 0; i < N; i++) {
    match &= G[i] == G_[i];
  }

  std::free(f);
  std::free(g);
  std::free(F);
  std::free(G);
  std::free(f_);
  std::free(g_);
  std::free(F_);
  std::free(G_);
  std::free(skey);

  assert(flg);
  assert(match);
}

}
