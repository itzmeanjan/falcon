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

}
