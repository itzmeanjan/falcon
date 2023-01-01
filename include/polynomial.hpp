#pragma once
#include "ff.hpp"
#include "fft.hpp"
#include "ntt.hpp"

// Polynomial arithmetic over Falcon Prime Field Z_q | q = 3 * (2 ^ 12) + 1
namespace polynomial {

// Multiply two degree-{(1 << lg2n) - 1} polynomials in their FFT form, by
// performing element-wise multiplication over C
template<const size_t lg2n>
inline void
mul(const fft::cmplx* const __restrict polya,
    const fft::cmplx* const __restrict polyb,
    fft::cmplx* const __restrict polyc)
{
  constexpr size_t n = 1ul << lg2n;

  for (size_t i = 0; i < n; i++) {
    polyc[i] = polya[i] * polyb[i];
  }
}

// Multiply two degree-{(1 << lg2n) - 1} polynomials in their NTT form, by
// performing element-wise multiplication over Z_q
template<const size_t lg2n>
inline void
mul(const ff::ff_t* const __restrict polya,
    const ff::ff_t* const __restrict polyb,
    ff::ff_t* const __restrict polyc)
{
  constexpr size_t n = 1ul << lg2n;

  for (size_t i = 0; i < n; i++) {
    polyc[i] = polya[i] * polyb[i];
  }
}

// Divide one degree-{(1 << lg2n) - 1} polynomial by another one, in their NTT
// form, by performing element-wise division over Z_q
//
// Note, because multiplicative inverse of additive identity element ( i.e. 0 )
// can't be computed, attempt to divide by 0 over Z_q, should result in 0. As
// this implementation doesn't emit any kind of exceptions, it might be little
// problematic when using these APIs.
template<const size_t lg2n>
inline void
div(const ff::ff_t* const __restrict polya,
    const ff::ff_t* const __restrict polyb,
    ff::ff_t* const __restrict polyc)
{
  constexpr size_t n = 1ul << lg2n;

  for (size_t i = 0; i < n; i++) {
    polyc[i] = polya[i] / polyb[i];
  }
}

}
