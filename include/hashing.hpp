#pragma once
#include "ff.hpp"
#include "shake256.hpp"

// Message Hashing for Falcon-{512, 1024}
namespace hashing {

// Given uniform random sampled salt bytes ( of length `slen` ) and message of
// length `mlen` bytes, this function first absorbs salt and message ( in order
// ) into SHAKE256 XOF state and then computes a degree-(n-1) polynomial over
// Z_q | q = 12289, by squeezing bytes out of keccak sponge state and rejection
// sampling.
//
// This function is the implementation of algorithm 3, described in section 3.7,
// on page 31 of Falcon specification https://falcon-sign.info/falcon.pdf
template<const size_t n>
inline void
hash_to_point(const uint8_t* const __restrict salt,
              const size_t slen,
              const uint8_t* const __restrict msg,
              const size_t mlen,
              ff::ff_t* const __restrict poly)
  requires((n == 512) || (n == 1024))
{
  constexpr size_t m = 1ul << 16;
  constexpr size_t q = ff::Q;
  constexpr size_t k = m / q;
  constexpr uint16_t kq = k * q;

  shake256::shake256<true> hasher{};
  hasher.absorb(salt, slen);
  hasher.absorb(msg, mlen);
  hasher.finalize();

  size_t coeff_idx = 0;
  uint8_t buf[shake256::rate >> 3];

  while (coeff_idx < n) {
    hasher.read(buf, sizeof(buf));

    for (size_t off = 0; (off < sizeof(buf)) && (coeff_idx < n); off += 2) {
      const uint16_t t = (static_cast<uint16_t>(buf[off + 0]) << 8) |
                         (static_cast<uint16_t>(buf[off + 1]) << 0);
      if (t < kq) {
        poly[coeff_idx] = ff::ff_t{ t };
        coeff_idx++;
      }
    }
  }
}

}
