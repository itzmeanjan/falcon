#pragma once
#include "ff.hpp"

// (inverse) Number Theoretic Transform for degree-511 polynomial, over
// Falcon Prime Field Z_q | q = 3 * (2 ^ 12) + 1
namespace ntt {

constexpr size_t LOG2N = 9;
constexpr size_t N = 1 << LOG2N;

// First primitive 1024 -th root of unity modulo q
//
// Meaning, 49 ** 1024 == 1 mod q
constexpr ff::ff_t ζ{ 49 };

// Multiplicative inverse of N over Z_q | N = 512
constexpr auto INV_N = ff::ff_t{ N }.inv();

// Given a 64 -bit unsigned integer, this routine extracts specified many
// contiguous bits from ( least significant bit ) LSB side & reverses their bit
// order, returning bit reversed `mbw` -bit wide number
//
// See
// https://github.com/itzmeanjan/dilithium/blob/776e4c35830cd330f59062a30b7c93ae6731e3a7/include/ntt.hpp#L56-L75
// for source of inspiration
template<const size_t mbw>
inline static constexpr size_t
bit_rev(const size_t v)
{
  size_t v_rev = 0ul;

  for (size_t i = 0; i < mbw; i++) {
    const size_t bit = (v >> i) & 0b1;
    v_rev ^= bit << (mbw - 1ul - i);
  }

  return v_rev;
}

// Given a polynomial f with 256 coefficients over Z_q | q = 2^23 - 2^13 + 1,
// this routine computes number theoretic transform using Cooley-Tukey
// algorithm, producing polynomial f' s.t. its coefficients are placed in
// bit-reversed order
//
// Note, this routine mutates input i.e. it's an in-place NTT implementation.
//
// Implementation inspired from
// https://github.com/itzmeanjan/dilithium/blob/776e4c35830cd330f59062a30b7c93ae6731e3a7/include/ntt.hpp#L77-L111
inline void
ntt(ff::ff_t* const __restrict poly)
{
  for (int64_t l = LOG2N - 1; l >= 0; l--) {
    const size_t len = 1ul << l;
    const size_t lenx2 = len << 1;
    const size_t k_beg = N >> (l + 1);

    for (size_t start = 0; start < N; start += lenx2) {
      const size_t k_now = k_beg + (start >> (l + 1));
      const ff::ff_t ζ_exp = ζ ^ bit_rev<LOG2N>(k_now);

      for (size_t i = start; i < start + len; i++) {
        const auto tmp = ζ_exp * poly[i + len];

        poly[i + len] = poly[i] - tmp;
        poly[i] = poly[i] + tmp;
      }
    }
  }
}

}
