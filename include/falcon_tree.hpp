#pragma once
#include "fft.hpp"
#include "polynomial.hpp"
#include <cstring>

// Construction of Falcon Tree from f, g, F, G ∈ Z[x]/(x^n + 1)
namespace falcon_tree {

// Given a full-rank self-adjoint matrix G = (G_ij) ∈ FFT(Q[x]/ φ)^(2×2), this
// routine computes LDL* decomposition of G = LDL* over FFT(Q[x]/ φ), following
// algorithm 8 of Falcon specification https://falcon-sign.info/falcon.pdf
template<const size_t N>
static inline void
ldl(const fft::cmplx* const __restrict G,
    fft::cmplx* const __restrict l10,
    fft::cmplx* const __restrict d00,
    fft::cmplx* const __restrict d11)
  requires((N > 1) && ((N & (N - 1)) == 0) && (N <= 1024))
{
  const fft::cmplx* g00 = G;
  const fft::cmplx* g10 = G + 2 * N;
  const fft::cmplx* g11 = G + 3 * N;

  std::memcpy(d00, g00, sizeof(fft::cmplx) * N);
  polynomial::div<log2<N>()>(g10, g00, l10);

  fft::cmplx tmp0[N];
  fft::cmplx tmp1[N];

  std::memcpy(tmp0, l10, sizeof(tmp0));
  fft::adj_poly<log2<N>()>(tmp0);
  polynomial::mul<log2<N>()>(l10, tmp0, tmp1);
  polynomial::mul<log2<N>()>(tmp1, g00, tmp0);
  polynomial::sub<log2<N>()>(g11, tmp0, d11);
}

}
