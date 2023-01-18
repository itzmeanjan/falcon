#pragma once
#include "polynomial.hpp"
#include "samplerz.hpp"

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
sqrd_norm(const double* const poly)
{
  double res = 0.;

  for (size_t i = 0; i < n; i++) {
    res += poly[i] * poly[i];
  }

  return res;
}

// Computes squared Gram-Schmidt norm of NTRU matrix generated using random
// sampled polynomials f, g of degree (N - 1) | N = 2^LOG2N
//
// This routine does what line 9 of algorithm 5 in the Falcon specification (
// https://falcon-sign.info/falcon.pdf ) does.
template<const size_t LOG2N>
static inline double
gram_schmidt_norm(const double* const __restrict f,
                  const double* const __restrict g)
{
  constexpr size_t N = 1ul << LOG2N;
  constexpr double q = ff::Q;
  constexpr double qxq = q * q;

  const double sq_norm_fg = sqrd_norm<N>(f) + sqrd_norm<N>(g);

  fft::cmplx f_[N];
  fft::cmplx g_[N];

  for (size_t i = 0; i < N; i++) {
    f_[i] = fft::cmplx{ f[i] };
    g_[i] = fft::cmplx{ g[i] };
  }

  fft::fft<LOG2N>(f_);
  fft::fft<LOG2N>(g_);

  fft::cmplx f_adj[N];
  fft::cmplx g_adj[N];

  std::memcpy(f_adj, f_, sizeof(f_));
  std::memcpy(g_adj, g_, sizeof(g_));

  fft::adj_poly<LOG2N>(f_adj);
  fft::adj_poly<LOG2N>(g_adj);

  fft::cmplx fxf_adj[N];
  fft::cmplx gxg_adj[N];

  polynomial::mul(f, f_adj, fxf_adj);
  polynomial::mul(g, g_adj, gxg_adj);

  fft::cmplx fxf_adj_gxg_adj[N];
  polynomial::add<LOG2N>(fxf_adj, gxg_adj, fxf_adj_gxg_adj);

  fft::cmplx ft[N];
  fft::cmplx gt[N];

  polynomial::div(f_adj, fxf_adj_gxg_adj, ft);
  polynomial::div(g_adj, fxf_adj_gxg_adj, gt);

  fft::ifft<LOG2N>(ft);
  fft::ifft<LOG2N>(gt);

  double ft_[N];
  double gt_[N];

  for (size_t i = 0; i < N; i++) {
    ft_[i] = std::real(ft[i]);
    gt_[i] = std::real(gt[i]);
  }

  const double sq_norm_FG = qxq * (sqrd_norm<N>(ft_) + sqrd_norm<N>(gt_));
  return std::max(sq_norm_fg, sq_norm_FG);
}

}
