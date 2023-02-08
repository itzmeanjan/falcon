#pragma once
#include "encoding.hpp"
#include "fft.hpp"
#include "keygen.hpp"
#include "signing.hpp"

// Falcon{512, 1024} Key Generation, Signing and Verification Algorithm
namespace falcon {

// [User Friendly API] Falcon{512, 1024} key generation algorithm, which takes
// no input, does following
//
// - Generates four random polynomials f, g, F, G s.t. it solves NTRU equation
// - Computes public key h = gf^-1 mod q
// - Serializes both public key and private key as byte arrays
//
// Note, this routine doesn't compute 2x2 matrix B = [[g, -f], [G, -F]] or
// Falcon Tree T, which are required during signing period. For those, see
// keygen.hpp file's keygen() function implementation - that is an
// implementation of algorithm 4 of Falcon specification, which does no byte
// serialization for secret key or public key.
template<const size_t N>
static inline void
keygen(uint8_t* const __restrict pkey, uint8_t* const __restrict skey)
  requires((N == 512) || (N == 1024))
{
  int32_t f[N];
  int32_t g[N];
  int32_t F[N];
  int32_t G[N];
  ff::ff_t h[N];

  ntru_gen::ntru_gen<N>(f, g, F, G);
  keygen::compute_public_key<N>(f, g, h);
  encoding::encode_pkey<N>(h, pkey);
  encoding::encode_skey<N>(f, g, F, skey);
}

// Given three degree N polynomials f, g and F, this routine recomputes G using
// NTRU equation fG - gF = q mod φ.
//
// This routine will be useful when secret key is loaded from disk ( which holds
// byte encapsulated value of polynomials f, g and F ) and G needs to be
// computed again because all of four polynomials f, g, F and G are required for
// computing Falcon Tree T, which is used for signing messages.
template<const size_t N>
static inline void
recompute_G(const int32_t* const __restrict f,
            const int32_t* const __restrict g,
            const int32_t* const __restrict F,
            int32_t* const __restrict G)
  requires((N == 512) || (N == 1024))
{
  constexpr double Q = ff::Q;

  fft::cmplx f_[N];
  fft::cmplx g_[N];
  fft::cmplx F_[N];
  fft::cmplx G_[N];
  fft::cmplx q[N];
  fft::cmplx tmp[N];

  for (size_t i = 0; i < N; i++) {
    f_[i] = fft::cmplx{ static_cast<double>(f[i]) };
    g_[i] = fft::cmplx{ static_cast<double>(g[i]) };
    F_[i] = fft::cmplx{ static_cast<double>(F[i]) };
    q[i] = fft::cmplx{ Q };
  }

  fft::fft<log2<N>()>(f_);
  fft::fft<log2<N>()>(g_);
  fft::fft<log2<N>()>(F_);

  polynomial::mul<log2<N>()>(g_, F_, tmp);
  polynomial::add_to<log2<N>()>(tmp, q);
  polynomial::div<log2<N>()>(tmp, f_, G_);

  fft::ifft<log2<N>()>(G_);

  for (size_t i = 0; i < N; i++) {
    G[i] = static_cast<int32_t>(std::round(G_[i].real()));
  }
}

// Given four degree N polynomials f, g, F and G, in coefficient form, this
// routine computes a 2x2 matrix B, in its FFT form s.t. B = [[g, -f], [G, -F]]
template<const size_t N>
static inline void
compute_matrix_B(const int32_t* const __restrict f,
                 const int32_t* const __restrict g,
                 const int32_t* const __restrict F,
                 const int32_t* const __restrict G,
                 fft::cmplx* const __restrict B)
  requires((N == 512) || (N == 1024))
{
  for (size_t i = 0; i < N; i++) {
    B[i] = fft::cmplx{ static_cast<double>(g[i]) };
    B[N + i] = fft::cmplx{ -static_cast<double>(f[i]) };
    B[2 * N + i] = fft::cmplx{ static_cast<double>(G[i]) };
    B[3 * N + i] = fft::cmplx{ -static_cast<double>(F[i]) };
  }

  fft::fft<log2<N>()>(B);
  fft::fft<log2<N>()>(B + N);
  fft::fft<log2<N>()>(B + 2 * N);
  fft::fft<log2<N>()>(B + 3 * N);
}

// Given a 2x2 matrix B ( in its FFT format ) s.t. B = [[g, -f], [G, -F]], this
// routine computes a falcon tree T, in its FFT format s.t. it takes (k+1) * 2^k
// -many complex numbers to store the full falcon tree when tree height is k =
// log2(N)
template<const size_t N>
static inline void
compute_falcon_tree(
  const fft::cmplx* const __restrict B, // 2x2 matrix [[g, -f], [G, -F]]
  fft::cmplx* const __restrict T        // Falcon Tree ( in FFT form )
  )
  requires((N == 512) || (N == 1024))
{
  // see table 3.3 of falcon specification
  constexpr double σ_values[]{ 165.736617183, 168.388571447 };
  constexpr double σ = σ_values[N == 1024];

  fft::cmplx gram_matrix[2 * 2 * N];
  keygen::compute_gram_matrix<N>(B, gram_matrix);

  falcon_tree::ffldl<N, 0, log2<N>()>(gram_matrix, T);
  falcon_tree::normalize_tree<N, 0, log2<N>()>(T, σ);
}

// Given a 2x2 matrix B ( in its FFT form ) s.t. B = [[g, -f], [G, -F]], falcon
// tree T ( in its FFT representation ) and message M of mlen -bytes, this
// routine computes a compressed Falcon{512, 1024} signature, following
// algorithm 10 of falcon specification.
//
// Note, compressed falcon signature doesn't include message bytes, looks like
//
// <1 -byte header> +
// <40 -bytes random salt> +
// <r -bytes compressed signature> s.t. r = (666 - 41) if N = 512
//                                      r = (1280 - 41) else if N = 1024
template<const size_t N>
static inline void
sign(const fft::cmplx* const __restrict B, // 2x2 matrix [[g, -f], [G, -F]]
     const fft::cmplx* const __restrict T, // Falcon Tree ( in FFT form )
     const uint8_t* const __restrict msg,  // message to be signed
     const size_t mlen,                    // = len(msg), in bytes
     uint8_t* const __restrict sig         // compressed falcon signature
     )
  requires((N == 512) || (N == 1024))
{
  constexpr int32_t β2_values[]{ 34034726, 70265242 };
  constexpr size_t slen_values[]{ 666, 1280 };
  constexpr double σ_min_values[]{ 1.277833697, 1.298280334 };

  constexpr int32_t β2 = β2_values[N == 1024];
  constexpr size_t slen = slen_values[N == 1024];
  constexpr double σ_min = σ_min_values[N == 1024];

  signing::sign<N, β2, slen>(B, T, msg, mlen, sig, σ_min);
}

}
