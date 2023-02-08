#pragma once
#include "ff.hpp"
#include "fft.hpp"
#include "ntt.hpp"

// Polynomial arithmetic over Falcon Prime Field Z_q | q = 3 * (2 ^ 12) + 1 and
// complex number field C
namespace polynomial {

// Add two degree-{(1 << lg2n) - 1} polynomials in their FFT form, by
// performing element-wise addition over C
template<const size_t lg2n>
inline void
add(const fft::cmplx* const __restrict polya,
    const fft::cmplx* const __restrict polyb,
    fft::cmplx* const __restrict polyc)
{
  constexpr size_t n = 1ul << lg2n;

  for (size_t i = 0; i < n; i++) {
    polyc[i] = polya[i] + polyb[i];
  }
}

// Accumulate one degree-{(1 << lg2n) - 1} polynomial into another one ( of same
// degree ), when both of them are in their FFT form, by performing element-wise
// addition over C
template<const size_t lg2n>
static inline void
add_to(fft::cmplx* const __restrict polya,
       const fft::cmplx* const __restrict polyb)
{
  constexpr size_t n = 1ul << lg2n;

  for (size_t i = 0; i < n; i++) {
    polya[i] += polyb[i];
  }
}

// Accumulate one degree-{(1 << lg2n) - 1} polynomial into another one ( of same
// degree ), when both of them are in their NTT form, by performing element-wise
// addition over Z_q
template<const size_t lg2n>
static inline void
add_to(ff::ff_t* const __restrict polya, const ff::ff_t* const __restrict polyb)
{
  constexpr size_t n = 1ul << lg2n;

  for (size_t i = 0; i < n; i++) {
    polya[i] += polyb[i];
  }
}

// Given a degree N polynomial ( in its NTT form ), this routine performs
// element wise negation over Z_q s.t. N = 2^log2n and q = 12289
template<const size_t log2n>
static inline void
neg(ff::ff_t* const __restrict poly)
{
  constexpr size_t n = 1ul << log2n;

  for (size_t i = 0; i < n; i++) {
    poly[i] = -poly[i];
  }
}

// Subtracts one degree-{(1 << lg2n) - 1} polynomial from another one, when both
// them are in their FFT form, by performing element-wise subtraction over C
template<const size_t lg2n>
inline void
sub(const fft::cmplx* const __restrict polya,
    const fft::cmplx* const __restrict polyb,
    fft::cmplx* const __restrict polyc)
{
  constexpr size_t n = 1ul << lg2n;

  for (size_t i = 0; i < n; i++) {
    polyc[i] = polya[i] - polyb[i];
  }
}

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

// Divide one degree-{(1 << lg2n) - 1} polynomial by another one, in their FFT
// form, by performing element-wise division over C
template<const size_t lg2n>
inline void
div(const fft::cmplx* const __restrict polya,
    const fft::cmplx* const __restrict polyb,
    fft::cmplx* const __restrict polyc)
{
  constexpr size_t n = 1ul << lg2n;

  for (size_t i = 0; i < n; i++) {
    polyc[i] = polya[i] / polyb[i];
  }
}

}
