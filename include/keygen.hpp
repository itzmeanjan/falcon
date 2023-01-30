#pragma once
#include "falcon_tree.hpp"

// Falcon{512, 1024} Key Pair Generation related Routines
namespace keygen {

// Given a matrix B of dimension 2x2 s.t. each element of matrix ∈ FFT(Q[x]/
// (x^N + 1)), this routine computes Gram matrix G = B x B*, following line 4 of
// algorithm 4 in Falcon specification.
//
// Note, each of 4 component polynomials of B, should be in their FFT form and
// resulting gram matrix G also has its components in FFT form.
//
// Computed gram marix G is passed to ffLDL* decomposition routine, which is
// used for computing Falcon tree T.
//
// See
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/ffsampling.py#L15-L31
// where it's shown how Gram matrix of B can be computed in coefficient
// representation.
template<const size_t N>
static inline void
compute_gram_matrix(
  const fft::cmplx* const __restrict B, // 2 x 2 x N complex numbers
  fft::cmplx* const __restrict G        // 2 x 2 x N complex numbers
  )
  requires((N > 1) && ((N & (N - 1)) == 0) && (N <= 1024))
{
  fft::cmplx B_adj[N * 2 * 2];
  fft::cmplx tmp[N];

  // compute B*
  fft::adj_poly<log2<N>()>(B_adj);
  fft::adj_poly<log2<N>()>(B_adj + N);
  fft::adj_poly<log2<N>()>(B_adj + 2 * N);
  fft::adj_poly<log2<N>()>(B_adj + 3 * N);

  // compute G[0][0]
  polynomial::mul<log2<N>()>(B, B_adj, G);
  polynomial::mul<log2<N>()>(B + N, B_adj + N, tmp);
  polynomial::add_to<log2<N>()>(G, tmp);

  // compute G[0][1]
  polynomial::mul<log2<N>()>(B, B_adj + 2 * N, G);
  polynomial::mul<log2<N>()>(B + N, B_adj + 3 * N, tmp);
  polynomial::add_to<log2<N>()>(G + N, tmp);

  // compute G[1][0]
  polynomial::mul<log2<N>()>(B + 2 * N, B_adj, G);
  polynomial::mul<log2<N>()>(B + 3 * N, B_adj + N, tmp);
  polynomial::add_to<log2<N>()>(G + 2 * N, tmp);

  // compute G[1][1]
  polynomial::mul<log2<N>()>(B + 2 * N, B_adj + 2 * N, G);
  polynomial::mul<log2<N>()>(B + 3 * N, B_adj + 3 * N, tmp);
  polynomial::add_to<log2<N>()>(G + 3 * N, tmp);
}

}
