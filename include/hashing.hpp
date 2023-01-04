#pragma once
#include "ff.hpp"
#include "shake256.hpp"

// Message Hashing for Falcon-{512, 1024}
namespace hashing {

// Given a message byte array of length `mlen` bytes, this function first
// absorbs it into SHAKE256 XOF state and then computes a degree-(n-1)
// polynomial over Z_q | q = 12289, by squeezing bytes out of keccak sponge
// state. Note, n ∈ {512, 1024}.
//
// This function is the implementation of algorithm 3, described in section 3.7,
// on page 31 of Falcon specification https://falcon-sign.info/falcon.pdf
template<const size_t n>
inline void
hash_to_point(const uint8_t* const __restrict msg,
              const size_t mlen,
              ff::ff_t* const __restrict poly)
{
  static_assert((n == 512) || (n == 1024), "N must ∈ {512, 1024}");

  constexpr size_t m = 1ul << 16;
  constexpr size_t q = ff::Q;
  constexpr size_t k = m / q;
  constexpr uint16_t kq = k * q;

  shake256::shake256 hasher{};
  hasher.hash(msg, mlen);

  size_t i = 0;
  uint8_t buf[2];

  while (i < n) {
    hasher.read(buf, sizeof(buf));
    const uint16_t t = (static_cast<uint16_t>(buf[0]) << 8) |
                       (static_cast<uint16_t>(buf[1]) << 0);

    if (t < kq) {
      poly[i] = ff::ff_t{ t };
      i++;
    }
  }
}

}
