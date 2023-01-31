#pragma once
#include "decoding.hpp"
#include "encoding.hpp"
#include "ntru_gen.hpp"
#include <cassert>

// Test functional correctness of Falcon PQC suite implementation
namespace test_falcon {

// Test whether random public key ( as polynomial over Fq | q = 12289 ), can be
// correctly encoded/ decoded or not.
template<const size_t N>
void
test_encoding_pkey()
{
  constexpr size_t pklen = falcon_utils::compute_pkey_len<N>();

  auto h = static_cast<ff::ff_t*>(std::malloc(sizeof(ff::ff_t) * N));
  auto pkey = static_cast<uint8_t*>(std::malloc(pklen));
  auto h_ = static_cast<ff::ff_t*>(std::malloc(sizeof(ff::ff_t) * N));

  for (size_t i = 0; i < N; i++) {
    h[i] = ff::ff_t::random();
  }

  encoding::encode_pkey<N>(h, pkey);
  const bool flg = decoding::decode_pkey<N>(pkey, h_);

  bool success = true;
  for (size_t i = 0; i < N; i++) {
    success &= h[i] == h_[i];
  }

  std::free(h);
  std::free(pkey);
  std::free(h_);

  assert(flg);
  assert(success);
}

// Test whether randomly generated ( using NTRUGen ) Falcon secret key can be
// correctly encoded/ decoded or not.
template<const size_t N>
void
test_encoding_skey()
{
  constexpr size_t sklen = falcon_utils::compute_skey_len<N>();

  auto f = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto g = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto F = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto G = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto skey = static_cast<uint8_t*>(std::malloc(sklen));
  auto f_ = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto g_ = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto F_ = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));

  ntru_gen::ntru_gen<N>(f, g, F, G);
  encoding::encode_skey<N>(f, g, F, skey);
  const bool flg = decoding::decode_skey<N>(skey, f_, g_, F_);

  bool success = true;
  for (size_t i = 0; i < N; i++) {
    success &= f[i] == f_[i];
    success &= g[i] == g_[i];
    success &= F[i] == F_[i];
  }

  std::free(f);
  std::free(g);
  std::free(F);
  std::free(G);
  std::free(skey);
  std::free(f_);
  std::free(g_);
  std::free(F_);

  assert(flg);
  assert(success);
}

}
