#pragma once
#include "falcon_tree.hpp"
#include "polynomial.hpp"
#include "samplerz.hpp"

// Fast Fourier Sampling
namespace ffsampling {

// Given two polynomials t0, t1 ∈ FFT(Q[x]/ (x^N + 1)) i.e. in their FFT
// representation and Falcon Tree T ( in its FFT representation ), this routine
// computes two polynomials z0, z1 ∈ FFT (Z[x]/ (x^N + 1)), using algorithm 11 (
// ffSampling ) as defined in Falcon specification
// https://falcon-sign.info/falcon.pdf
//
// For understanding ffSampling, you should read section 3.9 of specification.
template<const size_t N, const size_t AT_LEVEL, const size_t T_HEIGHT>
static inline void
ff_sampling(const fft::cmplx* const __restrict t0,
            const fft::cmplx* const __restrict t1,
            const fft::cmplx* const __restrict T,
            const double σ_min,
            fft::cmplx* const __restrict z0,
            fft::cmplx* const __restrict z1)
  requires((N > 0) && ((N & (N - 1)) == 0) && (N <= 1024) &&
           (AT_LEVEL <= T_HEIGHT) && (N == (1ul << (T_HEIGHT - AT_LEVEL))))
{
  constexpr size_t node_cnt = 1ul << AT_LEVEL;
  constexpr size_t tree_off = node_cnt * N;

  if constexpr (N == 1) {
    // deepest level of recursion !
    static_assert(AT_LEVEL == T_HEIGHT, "Can't go below leaf level of tree !");

    const double σ_prime = T[0].real();
    const auto z0_ = samplerz::samplerz(t0[0].real(), σ_prime, σ_min);
    const auto z1_ = samplerz::samplerz(t1[0].real(), σ_prime, σ_min);

    z0[0] = fft::cmplx{ static_cast<double>(z0_) };
    z1[0] = fft::cmplx{ static_cast<double>(z1_) };

    return;
  } else {
    static_assert(AT_LEVEL < T_HEIGHT, "Can go to leaf level !");

    const auto l = T;
    const auto Tl = T + tree_off;
    const auto Tr = Tl + (N / 2);
    const auto z0l = z0;
    const auto z1l = z1;
    const auto z0r = z0l + (N / 2);
    const auto z1r = z1l + (N / 2);

    fft::cmplx t1_0[N / 2];
    fft::cmplx t1_1[N / 2];

    fft::split_fft<log2<N>()>(t1, t1_0, t1_1);
    ff_sampling<N / 2, AT_LEVEL + 1, T_HEIGHT>(t1_0, t1_1, Tr, σ_min, z0r, z1r);

    fft::cmplx merged_z1[N];
    fft::merge_fft<log2<N / 2>()>(z0r, z1r, merged_z1);

    fft::cmplx tmp0[N];
    fft::cmplx tmp1[N];
    polynomial::sub<log2<N>()>(t1, merged_z1, tmp0);
    polynomial::mul<log2<N>()>(tmp0, l, tmp1);
    polynomial::add<log2<N>()>(t0, tmp1, tmp0);

    // t0' = tmp0

    fft::cmplx t0_0[N / 2];
    fft::cmplx t0_1[N / 2];

    fft::split_fft<log2<N>()>(tmp0, t0_0, t0_1);
    ff_sampling<N / 2, AT_LEVEL + 1, T_HEIGHT>(t0_0, t0_1, Tl, σ_min, z0l, z1l);

    fft::cmplx merged_z0[N];
    fft::merge_fft<log2<N / 2>()>(z0l, z1l, merged_z0);

    std::memcpy(z0, merged_z0, sizeof(merged_z0));
    std::memcpy(z1, merged_z1, sizeof(merged_z1));

    return;
  }
}

}
