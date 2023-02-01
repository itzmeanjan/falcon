#pragma once
#include "encoding.hpp"
#include "keygen.hpp"

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

}
