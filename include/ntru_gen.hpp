#pragma once
#include "ff.hpp"
#include "ntt.hpp"
#include "samplerz.hpp"
#include <cstring>

// Generate f, g, F, G ∈ Z[x]/(φ) | fG − gF = q mod φ ( i.e. NTRU equation )
namespace ntru_gen {

// Generate a random polynomial of degree (n - 1) | n ∈ {512, 1024} and each
// coefficient is sampled from a gaussian distribution D_{Z, σ{f, g}, 0} with σ
// = 1.17 * √(q/ 8192) as described in equation 3.29 on page 34 of the Falcon
// specification https://falcon-sign.info/falcon.pdf
template<const size_t n>
static inline void
gen_poly(int32_t* const poly)
{
  constexpr double σ = 1.43300980528773;
  constexpr size_t k = 4096 / n;

  for (size_t i = 0; i < n; i++) {

    int32_t res = 0;
    for (size_t j = 0; j < k; j++) {
      if constexpr (n == 512) {
        res += samplerz::samplerz(0., σ, samplerz::FALCON512_σ_min);
      } else {
        res += samplerz::samplerz(0., σ, samplerz::FALCON1024_σ_min);
      }
    }

    poly[i] = res;
  }
}

// Given a polynomial of degree (n - 1) | n ∈ {512, 1024}, this routine checks
// whether it can be inverted by computing NTT representation of polynomial and
// ensuring none of the coefficients, in NTT representation, are zero.
template<const size_t n>
static inline bool
is_poly_invertible(const int32_t* const poly)
{
  constexpr int32_t q = ff::Q;
  ff::ff_t tmp[n];

  for (size_t i = 0; i < n; i++) {
    const bool flg = poly[i] < 0;
    tmp.v = static_cast<uint16_t>(flg * q + poly[i]);
  }

  if constexpr (n == 512) {
    ntt::ntt<9>(tmp);
  } else {
    ntt::ntt<10>(tmp);
  }

  bool flg = true;
  for (size_t i = 0; i < n; i++) {
    flg &= tmp[i].v != 0;
  }

  return flg;
}

// Given a polynomial of degree (n - 1) | n ∈ {512, 1024}, in its coefficient
// representation, this routine computes squared norm using formula 3.10, as
// described on top of page 24 of the Falcon specification
// https://falcon-sign.info/falcon.pdf
template<const size_t n>
static inline double
sqr_norm(const double* const poly)
{
  double res = 0.;

  for (size_t i = 0; i < n; i++) {
    res += poly[i] * poly[i];
  }

  return res;
}

}
