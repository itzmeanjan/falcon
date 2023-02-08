#pragma once
#include "common.hpp"
#include "encoding.hpp"
#include "ff.hpp"
#include "ffsampling.hpp"
#include "fft.hpp"
#include "hashing.hpp"
#include "ntru_gen.hpp"
#include "polynomial.hpp"
#include <cstring>

// Falcon{512, 1024} Signing related Routines
namespace signing {

// Given mlen -bytes message M, 2x2 matrix B ( in FFT format, holding Falcon
// secret key ) s.t. B = [[g, -f], [G, -F]] and falcon tree T ( in FFT format ),
// this routine attempts to sign message M, while sampling 40 -bytes random
// salt, from system randomness.
//
// Signature byte layout looks like:
//
// <1 -byte header> +
// <40 -bytes salt> +
// <remaining bytes holding compressed signature>
//
// This routine is an implementation of algorithm 10 of falcon specification
// https://falcon-sign.info/falcon.pdf s.t. it takes secret key ( as 2x2 matrix
// B ) and precomputed falcon tree as input.
template<const size_t N, const int32_t β2, const size_t slen>
static inline void
sign(const fft::cmplx* const __restrict B,
     const fft::cmplx* const __restrict T,
     const uint8_t* const __restrict msg,
     const size_t mlen,
     uint8_t* const __restrict sig,
     const double σ_min // see table 3.3 of falcon specification
     )
  requires(((N == 512) && (β2 == 34034726) && (slen == 666)) ||
           ((N == 1024) && (β2 == 70265242) && (slen == 1280)))
{
  constexpr uint8_t header = 0x30 | static_cast<uint8_t>(log2<N>());
  constexpr double β2_ = static_cast<double>(β2);

  uint8_t salt[40];
  random_fill(salt, sizeof(salt));

  ff::ff_t c[N];
  hashing::hash_to_point<N>(salt, sizeof(salt), msg, mlen, c);

  fft::cmplx c_fft[N];
  for (size_t i = 0; i < N; i++) {
    c_fft[i] = fft::cmplx{ static_cast<double>(c[i].v) };
  }
  fft::fft<log2<N>()>(c_fft);

  fft::cmplx t0[N];
  fft::cmplx t1[N];

  polynomial::mul<log2<N>()>(c_fft, B + 3 * N, t0);
  polynomial::mul<log2<N>()>(c_fft, B + N, t1);

  constexpr fft::cmplx q{ ff::Q };
  for (size_t i = 0; i < N; i++) {
    t0[i] /= q;
    t1[i] = -(t1[i] / q);
  }

  fft::cmplx z0[N];
  fft::cmplx z1[N];
  fft::cmplx tz0[N];
  fft::cmplx tz1[N];
  fft::cmplx s0[N];
  fft::cmplx s1[N];
  int32_t s2[N];
  fft::cmplx tmp[N];

  while (1) {
    // ffSampling i.e. compute z = (z0, z1), same as line 6 of algo 10
    ffsampling::ff_sampling<N, 0, log2<N>()>(t0, t1, T, σ_min, z0, z1);

    // compute tz = (tz0, tz1) = (t0 - z0, t1 - z1)
    polynomial::sub<log2<N>()>(t0, z0, tz0);
    polynomial::sub<log2<N>()>(t1, z1, tz1);

    // compute s = (s0, s1) = tz * B | tz is 1x2 and B = 2x2 ( of dimension )
    polynomial::mul<log2<N>()>(tz0, B, s0);
    polynomial::mul<log2<N>()>(tz1, B + 2 * N, tmp);
    polynomial::add_to<log2<N>()>(s0, tmp);

    polynomial::mul<log2<N>()>(tz0, B + N, s1);
    polynomial::mul<log2<N>()>(tz1, B + 3 * N, tmp);
    polynomial::add_to<log2<N>()>(s1, tmp);

    // compute (∥s0, s1∥) ^ 2
    const double sq_norm0 = ntru_gen::sqrd_norm<log2<N>()>(s0);
    const double sq_norm1 = ntru_gen::sqrd_norm<log2<N>()>(s1);
    const double sq_norm = sq_norm0 + sq_norm1;

    // check ∥s∥2 > ⌊β2⌋
    if (sq_norm <= β2_) {
      fft::ifft<log2<N>()>(s1);

      for (size_t i = 0; i < N; i++) {
        s2[i] = static_cast<int32_t>(std::round(s1[i].real()));
      }

      // check if signature has been compressed
      const bool compressed = encoding::compress_sig<N, slen>(s2, sig);
      if (compressed) {
        break;
      }
    }
  }

  sig[0] = header;
  std::memcpy(sig + 1, salt, sizeof(salt));
}

}
