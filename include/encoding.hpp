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

// Encodes three polynomials f, g and F ( note, G is not encoded into secret key
// ), as a byte array, following encoding format specified in section 3.11.5 of
// Falcon specification https://falcon-sign.info/falcon.pdf
template<const size_t N>
static inline void
encode_skey(const int32_t* const __restrict f,
            const int32_t* const __restrict g,
            const int32_t* const __restrict F,
            uint8_t* const __restrict skey)
  requires((N == 512) || (N == 1024))
{
  size_t skoff = 0;
  skey[skoff++] = 0x50 | static_cast<uint8_t>(log2<N>());

  if constexpr (N == 512) {
    // force compile-time branch evaluation
    static_assert(N == 512, "N must be = 512 !");
    constexpr int32_t wrap_at = 1 << 6;

    // coefficients of f use 6 -bits when N = 512 i.e. they ∈ [-31, 31]
    for (size_t foff = 0; foff < N; foff += 4, skoff += 3) {
      const bool flg0 = f[foff + 0] < 0;
      const bool flg1 = f[foff + 1] < 0;
      const bool flg2 = f[foff + 2] < 0;
      const bool flg3 = f[foff + 3] < 0;

      const uint8_t f0 = static_cast<uint8_t>(flg0 * wrap_at + f[foff + 0]);
      const uint8_t f1 = static_cast<uint8_t>(flg1 * wrap_at + f[foff + 1]);
      const uint8_t f2 = static_cast<uint8_t>(flg2 * wrap_at + f[foff + 2]);
      const uint8_t f3 = static_cast<uint8_t>(flg3 * wrap_at + f[foff + 3]);

      skey[skoff + 0] = (f1 << 6) | f0;
      skey[skoff + 1] = (f2 << 4) | (f1 >> 2);
      skey[skoff + 2] = (f3 << 2) | (f2 >> 4);
    }

    // same for g, coefficients use 6 -bits i.e. they ∈ [-31, 31]
    for (size_t goff = 0; goff < N; goff += 4, skoff += 3) {
      const bool flg0 = g[goff + 0] < 0;
      const bool flg1 = g[goff + 1] < 0;
      const bool flg2 = g[goff + 2] < 0;
      const bool flg3 = g[goff + 3] < 0;

      const uint8_t g0 = static_cast<uint8_t>(flg0 * wrap_at + g[goff + 0]);
      const uint8_t g1 = static_cast<uint8_t>(flg1 * wrap_at + g[goff + 1]);
      const uint8_t g2 = static_cast<uint8_t>(flg2 * wrap_at + g[goff + 2]);
      const uint8_t g3 = static_cast<uint8_t>(flg3 * wrap_at + g[goff + 3]);

      skey[skoff + 0] = (g1 << 6) | g0;
      skey[skoff + 1] = (g2 << 4) | (g1 >> 2);
      skey[skoff + 2] = (g3 << 2) | (g2 >> 4);
    }
  } else {
    // force compile-time branch evaluation
    static_assert(N == 1024, "N must be = 1024 !");
    constexpr int32_t wrap_at = 1 << 5;

    // coefficients of f use 5 -bits when N = 1024 i.e. they ∈ [-15, 15]
    for (size_t foff = 0; foff < N; foff += 8, skoff += 5) {
      const bool flg0 = f[foff + 0] < 0;
      const bool flg1 = f[foff + 1] < 0;
      const bool flg2 = f[foff + 2] < 0;
      const bool flg3 = f[foff + 3] < 0;
      const bool flg4 = f[foff + 4] < 0;
      const bool flg5 = f[foff + 5] < 0;
      const bool flg6 = f[foff + 6] < 0;
      const bool flg7 = f[foff + 7] < 0;

      const uint8_t f0 = static_cast<uint8_t>(flg0 * wrap_at + f[foff + 0]);
      const uint8_t f1 = static_cast<uint8_t>(flg1 * wrap_at + f[foff + 1]);
      const uint8_t f2 = static_cast<uint8_t>(flg2 * wrap_at + f[foff + 2]);
      const uint8_t f3 = static_cast<uint8_t>(flg3 * wrap_at + f[foff + 3]);
      const uint8_t f4 = static_cast<uint8_t>(flg4 * wrap_at + f[foff + 4]);
      const uint8_t f5 = static_cast<uint8_t>(flg5 * wrap_at + f[foff + 5]);
      const uint8_t f6 = static_cast<uint8_t>(flg6 * wrap_at + f[foff + 6]);
      const uint8_t f7 = static_cast<uint8_t>(flg7 * wrap_at + f[foff + 7]);

      skey[skoff + 0] = (f1 << 5) | f0;
      skey[skoff + 1] = (f3 << 7) | (f2 << 2) | (f1 >> 3);
      skey[skoff + 2] = (f4 << 4) | (f3 >> 1);
      skey[skoff + 3] = (f6 << 6) | (f5 << 1) | (f4 >> 4);
      skey[skoff + 4] = (f7 << 3) | (f6 >> 2);
    }

    // same for g, coefficients use 5 -bits i.e. they ∈ [-15, 15]
    for (size_t goff = 0; goff < N; goff += 8, skoff += 5) {
      const bool flg0 = g[goff + 0] < 0;
      const bool flg1 = g[goff + 1] < 0;
      const bool flg2 = g[goff + 2] < 0;
      const bool flg3 = g[goff + 3] < 0;
      const bool flg4 = g[goff + 4] < 0;
      const bool flg5 = g[goff + 5] < 0;
      const bool flg6 = g[goff + 6] < 0;
      const bool flg7 = g[goff + 7] < 0;

      const uint8_t g0 = static_cast<uint8_t>(flg0 * wrap_at + g[goff + 0]);
      const uint8_t g1 = static_cast<uint8_t>(flg1 * wrap_at + g[goff + 1]);
      const uint8_t g2 = static_cast<uint8_t>(flg2 * wrap_at + g[goff + 2]);
      const uint8_t g3 = static_cast<uint8_t>(flg3 * wrap_at + g[goff + 3]);
      const uint8_t g4 = static_cast<uint8_t>(flg4 * wrap_at + g[goff + 4]);
      const uint8_t g5 = static_cast<uint8_t>(flg5 * wrap_at + g[goff + 5]);
      const uint8_t g6 = static_cast<uint8_t>(flg6 * wrap_at + g[goff + 6]);
      const uint8_t g7 = static_cast<uint8_t>(flg7 * wrap_at + g[goff + 7]);

      skey[skoff + 0] = (g1 << 5) | g0;
      skey[skoff + 1] = (g3 << 7) | (g2 << 2) | (g1 >> 3);
      skey[skoff + 2] = (g4 << 4) | (g3 >> 1);
      skey[skoff + 3] = (g6 << 6) | (g5 << 1) | (g4 >> 4);
      skey[skoff + 4] = (g7 << 3) | (g6 >> 2);
    }
  }

  // coefficients of F always use 8 -bits i.e. they ∈ [-127, 127]
  for (size_t Foff = 0; Foff < N; Foff++, skoff++) {
    skey[skoff] = static_cast<uint8_t>(F[Foff]);
  }
}

}
