#pragma once
#include "common.hpp"
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

// Given a full-rank Gram matrix G ∈ FFT(Q[x]/ (x^N + 1))^(2×2), this routine
// computes Falcon tree T ( which is a binary tree ), by recursively splitting
// diagonal elements of D, which is obtained by repeated LDL* decomposition of
// G, in recursive manner, following algorithm 9 of Falcon specification
// https://falcon-sign.info/falcon.pdf.
//
// For understanding Falcon tree, you should look at figure 3.1 of specification
// and read bottom of page 26 of specification. Finally you should also go
// through section 3.8.3 of specification for understanding how it can be
// implemented.
//
// Note, falcon tree ( of height k ) being a binary tree, can be stored using (1
// + k) * 2^k complex numbers i.e. ensure that memory allocated under owner T
// has enough space for storing those many complex numbers. Also note, at
// deepest level of recursion i.e. when N = 2, only real part of complex number
// matters i.e. imaginary part is negligibly small.
template<const size_t N, const size_t AT_LEVEL, const size_t T_HEIGHT>
static inline void
ffldl(const fft::cmplx* const __restrict G, fft::cmplx* const __restrict T)
  requires((N > 1) && ((N & (N - 1)) == 0) && (N <= 1024) &&
           (AT_LEVEL < T_HEIGHT) && (N == (1ul << (T_HEIGHT - AT_LEVEL))))
{
  constexpr size_t node_cnt = 1ul << AT_LEVEL;
  constexpr size_t tree_off = node_cnt * N;

  fft::cmplx D00[N];
  fft::cmplx D11[N];

  ldl<N>(G, T, D00, D11);

  if constexpr (N == 2) {
    // deepest level of recursion !
    static_assert(AT_LEVEL == (T_HEIGHT - 1),
                  "Can't go below this level of tree !");

    std::memcpy(T + tree_off, D00, sizeof(D00) / 2);
    std::memcpy(T + tree_off + (N / 2), D11, sizeof(D11) / 2);

    return;
  } else {
    fft::cmplx d00[N / 2];
    fft::cmplx d01[N / 2];
    fft::cmplx d10[N / 2];
    fft::cmplx d11[N / 2];

    fft::split_fft<log2<N>()>(D00, d00, d01);
    fft::split_fft<log2<N>()>(D11, d10, d11);

    fft::cmplx G0[(N / 2) * 2 * 2];
    fft::cmplx G1[(N / 2) * 2 * 2];

    std::memcpy(G0, d00, sizeof(d00));
    std::memcpy(G0 + (N / 2), d01, sizeof(d01));
    std::memcpy(G0 + N, d01, sizeof(d01));
    std::memcpy(G0 + N + (N / 2), d00, sizeof(d00));
    fft::adj_poly<log2<N>()>(G0 + N);

    std::memcpy(G1, d10, sizeof(d10));
    std::memcpy(G1 + (N / 2), d11, sizeof(d11));
    std::memcpy(G1 + N, d11, sizeof(d11));
    std::memcpy(G1 + N + (N / 2), d10, sizeof(d10));
    fft::adj_poly<log2<N>()>(G1 + N);

    ffldl<N / 2, AT_LEVEL + 1, T_HEIGHT>(G0, T + tree_off);
    ffldl<N / 2, AT_LEVEL + 1, T_HEIGHT>(G1, T + tree_off + (N / 2));

    return;
  }
}

}
