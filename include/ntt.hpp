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

}
