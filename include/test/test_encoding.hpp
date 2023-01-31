#pragma once
#include "decoding.hpp"
#include "encoding.hpp"
#include <cassert>

// Test functional correctness of Falcon PQC suite implementation
namespace test_falcon {

// Test whether random public key ( as polynomial over Fq | q = 12289 ), can be
// correctly encoded and decoded or not.
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

}
