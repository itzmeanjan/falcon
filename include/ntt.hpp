#pragma once
#include "ff.hpp"

// (inverse) Number Theoretic Transform for degree-{511, 1023} polynomial, over
// Falcon Prime Field Z_q | q = 3 * (2 ^ 12) + 1
namespace ntt {

constexpr size_t FALCON512_LOG2N = 9;
constexpr size_t FALCON512_N = 1ul << FALCON512_LOG2N;

constexpr size_t FALCON1024_LOG2N = 10;
constexpr size_t FALCON1024_N = 1ul << FALCON1024_LOG2N;

// First primitive 1024 -th root of unity modulo q
//
// Meaning, 49 ** 1024 == 1 mod q
constexpr ff::ff_t FALCON512_ζ{ 49 };

// First primitive 2048 -th root of unity modulo q
//
// Meaning, 7 ** 2048 == 1 mod q
constexpr ff::ff_t FALCON1024_ζ{ 7 };

// Multiplicative inverse of N over Z_q | N = 512
constexpr auto INV_FALCON512_N = ff::ff_t{ FALCON512_N }.inv();

// Multiplicative inverse of N over Z_q | N = 1024
constexpr auto INV_FALCON1024_N = ff::ff_t{ FALCON1024_N }.inv();

// Compile-time check to ensure that we're working with N ∈ {512, 1024}.
consteval bool
check_log2n(const size_t lgn)
{
  return (lgn == FALCON512_LOG2N) || (lgn == FALCON1024_LOG2N);
}

// Given a 64 -bit unsigned integer, this routine extracts specified many
// contiguous bits from ( least significant bit ) LSB side & reverses their bit
// order, returning bit reversed `mbw` -bit wide number
//
// See
// https://github.com/itzmeanjan/dilithium/blob/776e4c3/include/ntt.hpp#L56-L75
// for source of inspiration
template<const size_t mbw>
inline static constexpr size_t
bit_rev(const size_t v)
  requires(check_log2n(mbw))
{
  size_t v_rev = 0ul;

  for (size_t i = 0; i < mbw; i++) {
    const size_t bit = (v >> i) & 0b1;
    v_rev ^= bit << (mbw - 1ul - i);
  }

  return v_rev;
}

// Given a polynomial f with {512, 1024} coefficients s.t. each coefficient ∈
// Z_q, this routine computes number theoretic transform using Cooley-Tukey
// algorithm, producing {512, 1024} evaluations f' s.t. they are placed in
// bit-reversed order.
//
// Note, this routine mutates input i.e. it's an in-place NTT implementation.
//
// Implementation inspired from
// https://github.com/itzmeanjan/dilithium/blob/776e4c3/include/ntt.hpp#L77-L111
template<const size_t LOG2N>
inline void
ntt(ff::ff_t* const __restrict poly)
  requires(check_log2n(LOG2N))
{
  constexpr size_t N = 1ul << LOG2N;

  for (int64_t l = LOG2N - 1; l >= 0; l--) {
    const size_t len = 1ul << l;
    const size_t lenx2 = len << 1;
    const size_t k_beg = N >> (l + 1);

    for (size_t start = 0; start < N; start += lenx2) {
      const size_t k_now = k_beg + (start >> (l + 1));
      ff::ff_t ζ_exp{};

      if constexpr (LOG2N == FALCON512_LOG2N) {
        ζ_exp = FALCON512_ζ ^ bit_rev<LOG2N>(k_now);
      } else {
        ζ_exp = FALCON1024_ζ ^ bit_rev<LOG2N>(k_now);
      }

      for (size_t i = start; i < start + len; i++) {
        const auto tmp = ζ_exp * poly[i + len];

        poly[i + len] = poly[i] - tmp;
        poly[i] = poly[i] + tmp;
      }
    }
  }
}

// Given {512, 1024} evaluations of polynomial f s.t. each evaluation ∈ Z_q and
// they are placed in bit-reversed order, this routine computes inverse number
// theoretic transform using Gentleman-Sande algorithm, producing polynomial f'
// s.t. its {512, 1024} coefficients are placed in standard order.
//
// Note, this routine mutates input i.e. it's an in-place iNTT implementation.
//
// Implementation inspired from
// https://github.com/itzmeanjan/dilithium/blob/776e4c3/include/ntt.hpp#L113-L150
template<const size_t LOG2N>
inline void
intt(ff::ff_t* const __restrict poly)
  requires(check_log2n(LOG2N))
{
  constexpr size_t N = 1ul << LOG2N;

  for (size_t l = 0; l < LOG2N; l++) {
    const size_t len = 1ul << l;
    const size_t lenx2 = len << 1;
    const size_t k_beg = (N >> l) - 1;

    for (size_t start = 0; start < N; start += lenx2) {
      const size_t k_now = k_beg - (start >> (l + 1));
      ff::ff_t neg_ζ_exp{};

      if constexpr (LOG2N == FALCON512_LOG2N) {
        neg_ζ_exp = -(FALCON512_ζ ^ bit_rev<LOG2N>(k_now));
      } else {
        neg_ζ_exp = -(FALCON1024_ζ ^ bit_rev<LOG2N>(k_now));
      }

      for (size_t i = start; i < start + len; i++) {
        const auto tmp = poly[i];

        poly[i] = poly[i] + poly[i + len];
        poly[i + len] = tmp - poly[i + len];
        poly[i + len] = poly[i + len] * neg_ζ_exp;
      }
    }
  }

  for (size_t i = 0; i < N; i++) {
    if constexpr (LOG2N == FALCON512_LOG2N) {
      poly[i] = poly[i] * INV_FALCON512_N;
    } else {
      poly[i] = poly[i] * INV_FALCON1024_N;
    }
  }
}

}
