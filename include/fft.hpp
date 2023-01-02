#pragma once
#include "ntt.hpp"
#include <cmath>
#include <complex>
#include <numbers>

// (inverse) Fast Fourier Transform of degree-{511, 1023} polynomial f ∈
// Q[x]/(φ) s.t. φ is a monic with distinct roots over C
namespace fft {

using cmplx = std::complex<double>;

// Given k ∈ [0, n), this routine computes e ^ (i * ((π * k) / n)) using Euler's
// formula https://en.wikipedia.org/wiki/Euler%27s_formula
//
// e ^ iθ = cosθ + isinθ | i = √-1
template<const size_t n>
inline cmplx
computeζ(const size_t k)
{
  constexpr double π = std::numbers::pi;
  const double θ = π * static_cast<double>(k) / static_cast<double>(n);
  return { std::cos(θ), std::sin(θ) };
}

// Given a polynomial f ∈ Q[x]/(φ) with {512, 1024} coefficients, this routine
// computes fast fourier transform using Cooley-Tukey algorithm, producing
// {512, 1024} evaluations of f s.t. they are placed in bit-reversed order.
//
// Note, this routine mutates input i.e. it's an in-place FFT implementation.
//
// Implementation inspired from
// https://github.com/itzmeanjan/falcon/blob/4ab9f60/include/ntt.hpp#L59-L98
template<const size_t LOG2N>
inline void
fft(cmplx* const __restrict vec)
  requires(ntt::check_log2n(LOG2N))
{
  constexpr size_t N = 1ul << LOG2N;

  for (int64_t l = LOG2N - 1; l >= 0; l--) {
    const size_t len = 1ul << l;
    const size_t lenx2 = len << 1;
    const size_t k_beg = N >> (l + 1);

    for (size_t start = 0; start < N; start += lenx2) {
      const size_t k_now = k_beg + (start >> (l + 1));
      const auto ζ_exp = computeζ<N>(ntt::bit_rev<LOG2N>(k_now));

      for (size_t i = start; i < start + len; i++) {
        const auto tmp = ζ_exp * vec[i + len];

        vec[i + len] = vec[i] - tmp;
        vec[i] = vec[i] + tmp;
      }
    }
  }
}

// Given {512, 1024} evaluations of polynomial f ∈ Q[x]/(φ) s.t. each evaluation
// ∈ C and they are placed in bit-reversed order, this routine computes inverse
// fast fourier transform using Gentleman-Sande algorithm, producing polynomial
// f s.t. its {512, 1024} coefficients are placed in standard order.
//
// Note, this routine mutates input i.e. it's an in-place iFFT implementation.
//
// Implementation inspired from
// https://github.com/itzmeanjan/falcon/blob/4ab9f60/include/ntt.hpp#L59-L98
template<const size_t LOG2N>
inline void
ifft(cmplx* const __restrict vec)
  requires(ntt::check_log2n(LOG2N))
{
  constexpr size_t N = 1ul << LOG2N;
  constexpr double INV_N = 1. / static_cast<double>(N);

  for (size_t l = 0; l < LOG2N; l++) {
    const size_t len = 1ul << l;
    const size_t lenx2 = len << 1;
    const size_t k_beg = (N >> l) - 1;

    for (size_t start = 0; start < N; start += lenx2) {
      const size_t k_now = k_beg - (start >> (l + 1));
      const auto neg_ζ_exp = -computeζ<N>(ntt::bit_rev<LOG2N>(k_now));

      for (size_t i = start; i < start + len; i++) {
        const auto tmp = vec[i];

        vec[i] = vec[i] + vec[i + len];
        vec[i + len] = tmp - vec[i + len];
        vec[i + len] = vec[i + len] * neg_ζ_exp;
      }
    }
  }

  for (size_t i = 0; i < N; i++) {
    vec[i] = vec[i] * INV_N;
  }
}

// Splits a polynomial f into two polynomials f0, f1 s.t. all the polynomials
// are in their FFT representation.
//
// This routine is an implementation of the algorithm 1, described on page 29 of
// Falcon specification https://falcon-sign.info/falcon.pdf
template<const size_t LOG2N>
inline void
split_fft(const cmplx* const __restrict f,
          cmplx* const __restrict f0,
          cmplx* const __restrict f1)
{
  static_assert(LOG2N > 0, "Input vector length must be >= 2");

  constexpr size_t N = 1ul << LOG2N;
  constexpr size_t hN = N >> 1;
  constexpr size_t qN = hN >> 1;

  if constexpr (LOG2N == 1) {
    const auto ζ_exp = computeζ<N>(ntt::bit_rev<LOG2N>(1));

    f0[0] = 0.5 * (f[0] + f[1]);
    f1[0] = 0.5 * (f[0] - f[1]) * std::conj(ζ_exp);
  } else {
    for (size_t i = 0; i < hN; i++) {
      f0[i] = 0.5 * (f[2 * i] + f[2 * i + 1]);

      if (i < qN) {
        const auto ζ_exp = computeζ<N>(ntt::bit_rev<LOG2N>(hN + i * 2));
        const cmplx br[]{ std::conj(ζ_exp), ζ_exp };

        f1[i] = 0.5 * (f[2 * i] - f[2 * i + 1]) * br[i & 0b1ul];
      } else {
        const auto ζ_exp = computeζ<N>(ntt::bit_rev<LOG2N>(hN + (i - qN) * 2));
        const cmplx br[]{ ζ_exp, std::conj(ζ_exp) };

        f1[i] = 0.5 * (f[2 * i] - f[2 * i + 1]) * br[(i - qN) & 0b1ul];
      }
    }
  }
}

}
