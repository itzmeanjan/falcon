#pragma once
#include "decoding.hpp"
#include "ff.hpp"
#include "hashing.hpp"
#include "ntt.hpp"
#include "polynomial.hpp"

// Falcon{512, 1024} Signature Verification related Routines
namespace verification {

// Given mlen -bytes message, {666, 1280} -bytes signature ( encapsulating
// polynomial s2 ) and Falcon{512, 1024} public key as degree N polynomial over
// Z_q ( i.e. h ), this routine checks whether s1 + s2*h = c ( mod q ) equation
// holds or not, by computing s1, using arithmetic over Z_q[x]/(x^N + 1) and
// trying to assert if squared norm of vector of polynomials (s1, s2) is within
// expected bound β2.
//
// This routine returns boolean truth value in case of successful signature
// verification, otherwise it returns false.
template<const size_t N, const int32_t β2>
static inline bool
verify(const ff::ff_t* const __restrict h,
       const uint8_t* const __restrict msg,
       const size_t mlen,
       const uint8_t* const __restrict sig)
  requires((N == 512) || (N == 1024))
{
  uint8_t salt[40];
  int32_t s2[N];

  const size_t decoded = decoding::decode_sig<N>(sig, salt, s2);
  if (!decoded) [[unlikely]] {
    return decoded;
  }

  ff::ff_t s2_ntt[N];
  for (size_t i = 0; i < N; i++) {
    s2_ntt[i].v = static_cast<uint16_t>((s2[i] < 0) * ff::Q + s2[i]);
  }

  ff::ff_t c[N];
  hashing::hash_to_point<N>(salt, sizeof(salt), msg, mlen, c);

  ff::ff_t h_[N];
  std::memcpy(h_, h, sizeof(h_));

  ntt::ntt<log2<N>()>(c);
  ntt::ntt<log2<N>()>(s2_ntt);
  ntt::ntt<log2<N>()>(h_);

  ff::ff_t s1[N];

  polynomial::mul<log2<N>()>(s2_ntt, h_, s1); // s1 <- s2 * h ( mod q ) [NTT]
  polynomial::neg<log2<N>()>(s1);             // s1 <- -s1 ( mod q ) [NTT]
  polynomial::add_to<log2<N>()>(s1, c);       // s1 <- s1 + c ( mod q ) [NTT]

  ntt::intt<log2<N>()>(s1); // s1 <- c - s2*h ( mod q ) [Coeff]

  constexpr uint16_t qby2 = ff::Q / 2;
  int32_t normalized_s1[N];

  for (size_t i = 0; i < N; i++) {
    const bool flg = s1[i].v >= qby2;
    const auto t0 = static_cast<int32_t>(s1[i].v);
    const auto t1 = static_cast<int32_t>(flg * ff::Q);

    normalized_s1[i] = t0 - t1;
  }

  // Accumulated in 64 bits on purpose. β2 fits in 32 bits, but the norms this test must REJECT do
  // not: for a signature that should fail, s1 = c - s2*h mod q is essentially uniform, so the sum
  // over N=512 terms reaches ~1.9e10 — nine times INT32_MAX. In a 32-bit accumulator that
  // overflows, and a wrapped negative value passes `<= β2`, so a signature over one message
  // verifies against another. Genuine signatures have a small norm and never overflow, which is
  // why round-trip tests and KATs do not see it.
  //
  // On the exact bound, since it looks off by one: the centring above uses `flg = (v >= qby2)`
  // with qby2 = Q/2 = 6144 by integer division, so v=6144 maps to -6145 and the range is
  // [-6145, 6143] — NOT the usual symmetric [-6144, 6144], which is what `if (v > q/2)` gives.
  // Hence 512 * 6145^2 = 19,333,644,800 rather than 512 * 6144^2 = 19,327,352,832. The conclusion
  // is the same either way; the asymmetry is noted only so the number is checkable. It cannot
  // change an accept/reject decision, because one coefficient of that size already squares to more
  // than β2 on its own.
  int64_t sqrd_norm = 0;

  for (size_t i = 0; i < N; i++) {
    const int64_t v = s2[i];
    sqrd_norm += v * v;
  }
  for (size_t i = 0; i < N; i++) {
    const int64_t v = normalized_s1[i];
    sqrd_norm += v * v;
  }

  return sqrd_norm <= β2;
}

}
