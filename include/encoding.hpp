#pragma once
#include "common.hpp"

// Falcon KeyPair and Signature Encoding Routines
namespace encoding {

// Encodes polynomial h of degree N, which is the Falcon public key, as a byte
// array, following encoding format described in section 3.11.4 of Falcon
// specification https://falcon-sign.info/falcon.pdf
template<const size_t N>
static inline void
encode_pkey(const ff::ff_t* const __restrict h, uint8_t* const __restrict pkey)
  requires((N == 512) || (N == 1024))
{
  constexpr uint8_t header = log2<N>();
  constexpr uint16_t mask6 = 0x3f;
  constexpr uint16_t mask4 = 0x0f;
  constexpr uint16_t mask2 = 0x03;

  pkey[0] = header;
  for (size_t hoff = 0, pkoff = 1; hoff < N; hoff += 4, pkoff += 7) {
    pkey[pkoff + 0] = static_cast<uint8_t>(h[hoff + 0].v);
    pkey[pkoff + 1] = (static_cast<uint8_t>(h[hoff + 1].v & mask2) << 6) |
                      (static_cast<uint8_t>(h[hoff + 0].v >> 8) << 0);
    pkey[pkoff + 2] = static_cast<uint8_t>(h[hoff + 1].v >> 2);
    pkey[pkoff + 3] = (static_cast<uint8_t>(h[hoff + 2].v & mask4) << 4) |
                      (static_cast<uint8_t>(h[hoff + 1].v >> 10) << 0);
    pkey[pkoff + 4] = static_cast<uint8_t>(h[hoff + 2].v >> 4);
    pkey[pkoff + 5] = (static_cast<uint8_t>(h[hoff + 3].v & mask6) << 2) |
                      (static_cast<uint8_t>(h[hoff + 2].v >> 12) << 0);
    pkey[pkoff + 6] = static_cast<uint8_t>(h[hoff + 3].v >> 6);
  }
}

}
