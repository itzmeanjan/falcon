#pragma once
#include "common.hpp"
#include "ff.hpp"
#include "utils.hpp"
#include <cassert>
#include <cstring>

// Falcon KeyPair and Signature Decoding Routines
namespace decoding {

// Decodes a byte encoded Falcon{512, 1024} public key as elements
// ∈ Fq | q = 12289, following section 3.11.4 of Falcon specification
// https://falcon-sign.info/falcon.pdf
//
// Note, Falcon public key's first byte i.e. header byte should be encoded as
// described in specification, otherwise decoding will fail, returning false. In
// case of successful decoding, truth value is returned.
template<const size_t N>
static inline bool
decode_pkey(const uint8_t* const __restrict pkey, ff::ff_t* const __restrict h)
  requires((N == 512) || (N == 1024))
{
  constexpr size_t pklen = falcon_utils::compute_pkey_len<N>();
  constexpr uint8_t header = log2<N>();
  constexpr uint8_t mask6 = 0x3f;
  constexpr uint8_t mask4 = 0x0f;
  constexpr uint8_t mask2 = 0x03;

  if (pkey[0] != header) [[unlikely]] {
    std::memset(h, 0, sizeof(ff::ff_t) * N);
    return false;
  }

  for (size_t pkoff = 1, hoff = 0; pkoff < pklen; pkoff += 7, hoff += 4) {
    h[hoff + 0].v = (static_cast<uint16_t>(pkey[pkoff + 1] & mask6) << 8) |
                    (static_cast<uint16_t>(pkey[pkoff + 0]) << 0);
    h[hoff + 1].v = (static_cast<uint16_t>(pkey[pkoff + 3] & mask4) << 10) |
                    (static_cast<uint16_t>(pkey[pkoff + 2]) << 2) |
                    (static_cast<uint16_t>(pkey[pkoff + 1] >> 6) << 0);
    h[hoff + 2].v = (static_cast<uint16_t>(pkey[pkoff + 5] & mask2) << 12) |
                    (static_cast<uint16_t>(pkey[pkoff + 4]) << 4) |
                    (static_cast<uint16_t>(pkey[pkoff + 3] >> 4) << 0);
    h[hoff + 3].v = (static_cast<uint16_t>(pkey[pkoff + 6]) << 6) |
                    (static_cast<uint16_t>(pkey[pkoff + 5] >> 2) << 0);
  }

  return true;
}

}
