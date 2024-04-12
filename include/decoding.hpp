#pragma once
#include "common.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cstring>
#include <iterator>

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
    std::fill_n(h, N, 0);
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
    std::fill_n(f, N, 0);
    std::fill_n(g, N, 0);
    std::fill_n(F, N, 0);

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

// Given a byte array, this routine extracts out 8 contiguous bits from given
// bit index ( bitoff ).
//
// Imagine we've a byte array of length N (>=2) s.t. bits inside each byte are
// enumerated as following
//
// b0,b1,b2,b3,b4,b5,b6,b7 | b0,b1,b2,b3,b4,b5,b6,b7 | ...
// ---------------------- | ----------------------- | ...
//      byte 0           |        byte 1           | ...
//
// b0 <- most significant bit
// b7 <- least significant bit
//
// Now think we're asked to extract out 8 contiguous bits. starting from bit
// index 5, then we'll figure
//
// - we're starting from byte index 0 ( = 5/ 8  )
// - we should start accessing from bit index 5 ( = 5% 8 ) of byte index 0
// - first we'll take 3 least significant bits of byte index 0
// - finally we'll take 5 most significant bits of byte index 1
// - that forms out uint8_t return value, holding 8 contiguous bits of interest
//
// If we're asked to extract 8 contigous bits, starting from bit index 11, then
// we should figure
//
// - start from byte index 1 ( = 11/ 8 )
// - start from bit index 3 ( = 11% 8 ) of byte index 1
// - take 5 least significant bits of byte index 1
// - and take 3 most significant bits of byte index 2
//
// Now notice, if we're working with a byte array of length 2 and we're asked to
// extract 8 contiguous bits, starting from bit index 11, we will access memory
// location which we're not supposed to be. So it's caller's responsibility to
// ensure that atleast 2 consecutive memory locations ( i.e. bytes ) can be
// accessed, starting from byte index (bitoff / 8).
static inline constexpr uint8_t
extract_8_contiguous_bits(const uint8_t* const __restrict bytes,
                          const size_t bitoff)
{
  const size_t byte_at = bitoff >> 3;
  const size_t bit_at = bitoff & 7ul;

  const uint16_t word = (static_cast<uint16_t>(bytes[byte_at]) << 8) |
                        static_cast<uint16_t>(bytes[byte_at + 1]);

  return static_cast<uint8_t>(word >> (8 - bit_at));
}

// Given a byte array and starting bit index, this routine extracts out next n
// contiguous bits s.t. n <= 8 and no bits from next byte is ever touched i.e.
// only bits living at starting byte index are extracted out.
//
// Lets take an example, say we've a byte array which looks like following, when
// bits are enumerated
//
// b0,b1,b2,b3,b4,b5,b6,b7 | b0,b1,b2,b3,b4,b5,b6,b7 / ...
// ---------------------- | ----------------------- / ...
//      byte 0           |        byte 1           / ...
//
// b0 <- most significant bit
// b7 <- least significant bit
//
// Now I want to extract all remaining bits starting from bit index 12, so I
// figure
//
// - starting byte index = 12/ 8 = 1
// - starting bit index inside byte index 1 = 12% 8 = 4
// - remaining bits in byte index 1 = 7 - 4 + 1 = 4
// - extracts out <b4,b5,b6,b7> of byte index 1 as <b4,b5,b6,b7,0,0,0,0>
static inline constexpr uint8_t
extract_rem_contiguous_bits_in_byte(const uint8_t* const __restrict bytes,
                                    const size_t bitoff)
{
  const size_t byte_at = bitoff >> 3;
  const size_t bit_at = bitoff & 7ul;

  return bytes[byte_at] << bit_at;
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
//
// In case of successful decompression, returns boolean truth value, otherwise
// returns false, denoting decompression failure.
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

  while ((coeff_idx < N) && (bit_idx < slen)) {
    int32_t coeff = 0;
    uint8_t sign_bit = 0;

    // extracts sign bit and low ( least significant ) 7 bits of coefficient
    {
      const uint8_t res = extract_8_contiguous_bits(sig, bit_idx);

      sign_bit = res >> 7; // sign bit
      coeff = res & 0x7f;  // low 7 bits of coefficient
      bit_idx += 8;
    }

    // extract high bits of coefficient, which was encoded using unary code
    {
      size_t k = std::countl_zero(extract_8_contiguous_bits(sig, bit_idx));
      if (k < 8) [[likely]] {
        coeff += (1 << 7) * k;
        bit_idx += k;
      } else {
        bit_idx += k;
        for (; bit_idx < slen;) {
          const auto ebits = std::min(8ul, slen - bit_idx);

          size_t v = 0;
          if (ebits < 8) {
            const auto t = extract_rem_contiguous_bits_in_byte(sig, bit_idx);
            v = std::countl_zero(t);
          } else {
            const auto t = extract_8_contiguous_bits(sig, bit_idx);
            v = std::countl_zero(t);
          }
          k += v;
          bit_idx += ebits;

          if (v < ebits) {
            break;
          }
        }
      }
    }

    // recompute coefficient s_i
    coeff = sign_bit == 1 ? -coeff : coeff;

    // enforce unique encoding of 0
    failed |= (coeff == 0) && (sign_bit == 1);
    if (failed) {
      break;
    }

    // seems all good with decoding of this coefficient
    poly_s[coeff_idx] = coeff;

    bit_idx += 1;
    coeff_idx += 1;
  }

  // enforce trailing bits are 0
  failed |= (bit_idx >= slen) | (coeff_idx < N);
  if (!failed) {
    for (; bit_idx < slen;) {
      const size_t ebits = std::min(8ul, slen - bit_idx);

      size_t v = 0;
      if (ebits == 8) [[likely]] {
        const auto t = extract_8_contiguous_bits(sig, bit_idx);
        v = std::countl_zero(t);
      } else {
        const auto t = extract_rem_contiguous_bits_in_byte(sig, bit_idx);
        v = std::countl_zero(t);
      }

      bit_idx += ebits;
      failed |= (v < ebits);
    }
  }

  std::memset(poly_s, 0, sizeof(int32_t) * N * failed);
  return !failed;
}

// Given a byte encoded ( and compressed ) Falcon{512, 1024} signature, this
// routine decodes it into 40 -bytes salt and degree N polynomial s2 ( by
// decompressing ).
//
// In case of successful signature decoding, this routine returns boolean truth
// value, otherwise it returns false.
template<const size_t N>
static inline bool
decode_sig(const uint8_t* const __restrict sig,
           uint8_t* const __restrict salt,
           int32_t* const __restrict s2)
  requires((N == 512) || (N == 1024))
{
  constexpr uint8_t header = 0x30 | static_cast<uint8_t>(log2<N>());
  constexpr size_t slen_values[]{ 666, 1280 };
  constexpr size_t slen = slen_values[N == 1024];

  if (sig[0] != header) [[unlikely]] {
    return false;
  }

  const bool decompressed = decompress_sig<N, slen>(sig, s2);
  if (!decompressed) [[unlikely]] {
    return false;
  }

  std::memcpy(salt, sig + 1, 40);
  return true;
}

}
