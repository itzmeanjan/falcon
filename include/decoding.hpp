#pragma once
#include "common.hpp"
#include "utils.hpp"
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

// Decodes a byte encoded Falcon{512, 1024} secret key into three polynomials f,
// g and F each of degree N s.t. each element ∈ [-6145, 6143], following
// section 3.11.5 of Falcon specification https://falcon-sign.info/falcon.pdf
//
// Note, Falcon secret key's first byte i.e. header byte should be encoded as
// described in specification, otherwise decoding will fail, returning false. In
// case of successful decoding, truth value is returned.
template<const size_t N>
static inline bool
decode_skey(const uint8_t* const __restrict skey,
            int32_t* const __restrict f,
            int32_t* const __restrict g,
            int32_t* const __restrict F)
  requires((N == 512) || (N == 1024))
{
  constexpr uint8_t header = 0x50 | static_cast<uint8_t>(log2<N>());

  if (skey[0] != header) [[unlikely]] {
    std::memset(f, 0, sizeof(int32_t) * N);
    std::memset(g, 0, sizeof(int32_t) * N);
    std::memset(F, 0, sizeof(int32_t) * N);

    return false;
  }

  size_t skoff = 1;

  if constexpr (N == 512) {
    // force compile-time branch evaluation
    static_assert(N == 512, "N must be = 512 !");

    constexpr uint8_t mask6 = 0x3f;
    constexpr uint8_t mask4 = 0x0f;
    constexpr uint8_t mask2 = 0x03;

    constexpr int32_t fg_wrap_at = 64;
    constexpr int32_t fg_max = (fg_wrap_at / 2) - 1;

    for (size_t foff = 0; foff < N; foff += 4, skoff += 3) {
      const int32_t f0 = skey[skoff] & mask6;
      const int32_t f1 = ((skey[skoff + 1] & mask4) << 2) | (skey[skoff] >> 6);
      int32_t f2 = ((skey[skoff + 2] & mask2) << 4) | (skey[skoff + 1] >> 4);
      const int32_t f3 = skey[skoff + 2] >> 2;

      f[foff + 0] = f0 - (f0 > fg_max) * fg_wrap_at;
      f[foff + 1] = f1 - (f1 > fg_max) * fg_wrap_at;
      f[foff + 2] = f2 - (f2 > fg_max) * fg_wrap_at;
      f[foff + 3] = f3 - (f3 > fg_max) * fg_wrap_at;
    }

    for (size_t goff = 0; goff < N; goff += 4, skoff += 3) {
      const int32_t g0 = skey[skoff] & mask6;
      const int32_t g1 = ((skey[skoff + 1] & mask4) << 2) | (skey[skoff] >> 6);
      int32_t g2 = ((skey[skoff + 2] & mask2) << 4) | (skey[skoff + 1] >> 4);
      const int32_t g3 = skey[skoff + 2] >> 2;

      g[goff + 0] = g0 - (g0 > fg_max) * fg_wrap_at;
      g[goff + 1] = g1 - (g1 > fg_max) * fg_wrap_at;
      g[goff + 2] = g2 - (g2 > fg_max) * fg_wrap_at;
      g[goff + 3] = g3 - (g3 > fg_max) * fg_wrap_at;
    }
  } else {
    // force compile-time branch evaluation
    static_assert(N == 1024, "N must be = 1024 !");

    constexpr uint8_t mask5 = 0x1f;
    constexpr uint8_t mask4 = 0x0f;
    constexpr uint8_t mask3 = 0x07;
    constexpr uint8_t mask2 = 0x03;
    constexpr uint8_t mask1 = 0x01;

    constexpr int32_t fg_wrap_at = 32;
    constexpr int32_t fg_max = (fg_wrap_at / 2) - 1;

    for (size_t foff = 0; foff < N; foff += 8, skoff += 5) {
      const int32_t f0 = skey[skoff] & mask5;
      const int32_t f1 = ((skey[skoff + 1] & mask2) << 3) | (skey[skoff] >> 5);
      const int32_t f2 = (skey[skoff + 1] >> 2) & mask5;
      int32_t f3 = ((skey[skoff + 2] & mask4) << 1) | (skey[skoff + 1] >> 7);
      int32_t f4 = ((skey[skoff + 3] & mask1) << 4) | (skey[skoff + 2] >> 4);
      const int32_t f5 = (skey[skoff + 3] >> 1) & mask5;
      int32_t f6 = ((skey[skoff + 4] & mask3) << 2) | (skey[skoff + 3] >> 6);
      const int32_t f7 = skey[skoff + 4] >> 3;

      f[foff + 0] = f0 - (f0 > fg_max) * fg_wrap_at;
      f[foff + 1] = f1 - (f1 > fg_max) * fg_wrap_at;
      f[foff + 2] = f2 - (f2 > fg_max) * fg_wrap_at;
      f[foff + 3] = f3 - (f3 > fg_max) * fg_wrap_at;
      f[foff + 4] = f4 - (f4 > fg_max) * fg_wrap_at;
      f[foff + 5] = f5 - (f5 > fg_max) * fg_wrap_at;
      f[foff + 6] = f6 - (f6 > fg_max) * fg_wrap_at;
      f[foff + 7] = f7 - (f7 > fg_max) * fg_wrap_at;
    }

    for (size_t goff = 0; goff < N; goff += 8, skoff += 5) {
      const int32_t g0 = skey[skoff] & mask5;
      const int32_t g1 = ((skey[skoff + 1] & mask2) << 3) | (skey[skoff] >> 5);
      const int32_t g2 = (skey[skoff + 1] >> 2) & mask5;
      int32_t g3 = ((skey[skoff + 2] & mask4) << 1) | (skey[skoff + 1] >> 7);
      int32_t g4 = ((skey[skoff + 3] & mask1) << 4) | (skey[skoff + 2] >> 4);
      const int32_t g5 = (skey[skoff + 3] >> 1) & mask5;
      int32_t g6 = ((skey[skoff + 4] & mask3) << 2) | (skey[skoff + 3] >> 6);
      const int32_t g7 = skey[skoff + 4] >> 3;

      g[goff + 0] = g0 - (g0 > fg_max) * fg_wrap_at;
      g[goff + 1] = g1 - (g1 > fg_max) * fg_wrap_at;
      g[goff + 2] = g2 - (g2 > fg_max) * fg_wrap_at;
      g[goff + 3] = g3 - (g3 > fg_max) * fg_wrap_at;
      g[goff + 4] = g4 - (g4 > fg_max) * fg_wrap_at;
      g[goff + 5] = g5 - (g5 > fg_max) * fg_wrap_at;
      g[goff + 6] = g6 - (g6 > fg_max) * fg_wrap_at;
      g[goff + 7] = g7 - (g7 > fg_max) * fg_wrap_at;
    }
  }

  constexpr int32_t F_wrap_at = 256;
  constexpr int32_t F_max = (F_wrap_at / 2) - 1;

  for (size_t Foff = 0; Foff < N; Foff++, skoff++) {
    F[Foff] = skey[skoff] - (skey[skoff] > F_max) * F_wrap_at;
  }

  return true;
}

// Given compressed signature bytes, this routine attempts to decompress it back
// to a degree N polynomial s.t. coefficients ∈ Z[x] and they are distributed
// around 0, using algorithm 18 of Falcon specification
// https://falcon-sign.info/falcon.pdf
//
// Layout of sig = <8 -bits of header> +
//                 <320 -bits of salt> +
//                 <{666, 1280} - 41 -bytes of compressed signature>
//
// This routine doesn't access first 41 -bytes of signature.
template<const size_t N, const size_t sbytelen>
static inline bool
decompress_sig(const uint8_t* const __restrict sig,
               int32_t* const __restrict poly_s)
  requires(((N == 512) && (sbytelen == 666)) ||
           ((N == 1024) && (sbytelen == 1280)))
{
  constexpr size_t slen = 8 * sbytelen; // signature bit length

  size_t bit_idx = 8 +  // header byte
                   320; // salt bytes
  size_t coeff_idx = 0;
  bool failed = false;

  while (coeff_idx < N) {
    int32_t coeff = 0;
    uint8_t sign_bit = 0;

    // extract sign bit
    {
      const size_t byte_idx = bit_idx >> 3;
      const size_t from_bit = bit_idx & 7ul;

      sign_bit = (sig[byte_idx] >> (7 - from_bit)) & 0b1;
      bit_idx += 1;
    }

    // extract next 7 bits, which are low bits of coefficient
    {
      for (size_t i = bit_idx; i < bit_idx + 7; i++) {
        const size_t byte_idx = i >> 3;
        const size_t from_bit = i & 7ul;

        const uint8_t bit = (sig[byte_idx] >> (7 - from_bit)) & 0b1;
        coeff |= static_cast<int32_t>(bit << (6 - (i - bit_idx)));
      }

      bit_idx += 7;
    }

    // extract high bits of coefficient, which was encoded using unary code
    {
      size_t k = 0;

      for (size_t i = bit_idx;; i++) {
        const size_t byte_idx = i >> 3;
        const size_t from_bit = i & 7ul;

        const uint8_t bit = (sig[byte_idx] >> (7 - from_bit)) & 0b1;
        if (bit != 0) {
          break;
        }

        k += 1;
      }

      coeff += (1 << 7) * k;
      bit_idx += k;
    }

    // recompute coefficient s_i
    coeff = sign_bit == 1 ? -coeff : coeff;

    // enforce unique encoding of 0
    failed |= (coeff == 0) && (sign_bit == 1);
    if (failed) {
      break;
    }

    // all good with decoding of this coefficient
    poly_s[coeff_idx] = coeff;

    bit_idx += 1;
    coeff_idx += 1;
  }

  // enforce trailing bits are 0
  failed |= bit_idx >= slen;
  if (!failed) {
    while (bit_idx < slen) {
      const size_t byte_idx = bit_idx >> 3;
      const size_t from_bit = bit_idx & 7ul;

      const uint8_t bit = (sig[byte_idx] >> (7 - from_bit)) & 0b1;
      failed |= (bit != 0);

      bit_idx += 1;
    }
  }

  std::memset(poly_s, 0, sizeof(int32_t) * N * failed);
  return !failed;
}

}
