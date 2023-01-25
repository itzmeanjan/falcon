#pragma once
#include "gmpxx.h"
#include "karatsuba.hpp"
#include "polynomial.hpp"
#include "samplerz.hpp"
#include <array>

// Generate f, g, F, G ∈ Z[x]/(φ) | fG − gF = q mod φ ( i.e. NTRU equation )
namespace ntru_gen {

// Generate a random polynomial of degree (n - 1) | n ∈ {512, 1024} and each
// coefficient is sampled from a gaussian distribution D_{Z, σ{f, g}, 0} with σ
// = 1.17 * √(q/ 8192) as described in equation 3.29 on page 34 of the Falcon
// specification https://falcon-sign.info/falcon.pdf
template<const size_t LOG2N>
static inline void
gen_poly(int32_t* const poly)
{
  constexpr size_t N = 1ul << LOG2N;
  constexpr size_t k = 4096 / N;

  constexpr double σ = 1.43300980528773;
  constexpr double br[]{ samplerz::FALCON512_σ_min,
                         samplerz::FALCON1024_σ_min };
  constexpr double σ_min = br[N == 1024];

  for (size_t i = 0; i < N; i++) {

    int32_t res = 0;
    for (size_t j = 0; j < k; j++) {
      res += samplerz::samplerz(0., σ, σ_min);
    }

    poly[i] = res;
  }
}

// Given a polynomial of degree (n - 1) | n ∈ {512, 1024}, this routine checks
// whether it can be inverted by computing NTT representation of polynomial and
// ensuring none of the coefficients, in NTT representation, are zero.
template<const size_t LOG2N>
static inline bool
is_poly_invertible(const int32_t* const poly)
{
  constexpr size_t N = 1ul << LOG2N;
  constexpr int32_t q = ff::Q;

  ff::ff_t tmp[N];

  for (size_t i = 0; i < N; i++) {
    const bool flg = poly[i] < 0;
    tmp.v = static_cast<uint16_t>(flg * q + poly[i]);
  }

  ntt::ntt<LOG2N>(tmp);

  bool flg = true;
  for (size_t i = 0; i < N; i++) {
    flg &= tmp[i].v != 0;
  }

  return flg;
}

// Given a polynomial of degree (n - 1) | n ∈ {512, 1024}, in its coefficient
// representation, this routine computes squared norm using formula 3.10, as
// described on top of page 24 of the Falcon specification
// https://falcon-sign.info/falcon.pdf
template<const size_t LOG2N>
static inline double
sqrd_norm(const double* const poly)
{
  constexpr size_t N = 1ul << LOG2N;
  double res = 0.;

  for (size_t i = 0; i < N; i++) {
    res += poly[i] * poly[i];
  }

  return res;
}

// Given a polynomial of degree (n - 1) | n ∈ {512, 1024}, in its FFT
// representation, this routine computes squared norm using formula 3.8, as
// described on top of page 24 of the Falcon specification
// https://falcon-sign.info/falcon.pdf
template<const size_t LOG2N>
static inline double
sqrd_norm(const fft::cmplx* const poly)
{
  constexpr size_t N = 1ul << LOG2N;
  constexpr double N_ = static_cast<double>(N);
  fft::cmplx res{};

  for (size_t i = 0; i < N; i++) {
    res += poly[i] * std::conj(poly[i]);
  }

  return std::real(res) / N_;
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

  const auto sq_norm_fg = sqrd_norm<LOG2N>(f) + sqrd_norm<LOG2N>(g);

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

  polynomial::mul<LOG2N>(f_, f_adj, fxf_adj);
  polynomial::mul<LOG2N>(g_, g_adj, gxg_adj);

  fft::cmplx fxf_adj_gxg_adj[N];
  polynomial::add<LOG2N>(fxf_adj, gxg_adj, fxf_adj_gxg_adj);

  fft::cmplx ft[N];
  fft::cmplx gt[N];

  polynomial::div<LOG2N>(f_adj, fxf_adj_gxg_adj, ft);
  polynomial::div<LOG2N>(g_adj, fxf_adj_gxg_adj, gt);

  const auto sq_norm_FG = qxq * (sqrd_norm<LOG2N>(ft) + sqrd_norm<LOG2N>(gt));
  return std::max(sq_norm_fg, sq_norm_FG);
}

// Computes field norm a polynomial ( in coefficient representation ) of degree
// N s.t. N > 1 and N = 2^i, projecting element of Q[x]/(x^n + 1) to
// Q[x]/(x^(n/2) + 1), following section 3.6.1 of the Falcon specification ( see
// bottom of page 30, formula 3.25 ) https://falcon-sign.info/falcon.pdf
//
// This implementation collects inspiration from
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/ntrugen.py#L61-L75
template<const size_t N>
static inline std::array<mpz_class, N / 2>
field_norm(const std::array<mpz_class, N>& poly)
  requires((N > 1) && ((N & (N - 1)) == 0))
{
  constexpr size_t Nby2 = N / 2;
  using nby2poly_t = std::array<mpz_class, Nby2>;

  nby2poly_t polye;
  nby2poly_t polyo;

  for (size_t i = 0; i < Nby2; i++) {
    polye[i] = poly[2 * i];
    polyo[i] = poly[2 * i + 1];
  }

  const nby2poly_t polye_sq = karatsuba::karamul(polye, polye);
  const nby2poly_t polyo_sq = karatsuba::karamul(polyo, polyo);

  nby2poly_t res = polye_sq;
  for (size_t i = 0; i < Nby2 - 1; i++) {
    res[i + 1] = res[i + 1] - polyo_sq[i];
  }
  res[0] = res[0] + polyo_sq[Nby2 - 1];

  return res;
}

// Uses extended GCD algorithm over Z, given x, y ∈ Z, computing a, b, g ∈ Z
// s.t. ax + by = g
//
// Adapts
// https://github.com/itzmeanjan/kyber/blob/3cd41a5/include/ff.hpp#L49-L82 for
// multi-precision integers
static inline std::array<mpz_class, 3>
xgcd(const mpz_class& x, const mpz_class& y)
{
  mpz_class old_r{ x }, r{ y };
  mpz_class old_s{ 1 }, s{ 0 };
  mpz_class old_t{ 0 }, t{ 1 };

  while (r != mpz_class{ 0 }) {
    mpz_class quotient = old_r / r;

    mpz_class tmp{ old_r };
    old_r = r;
    r = tmp - mpz_class(quotient * r);

    tmp = old_s;
    old_s = s;
    s = tmp - mpz_class(quotient * s);

    tmp = old_t;
    old_t = t;
    t = tmp - mpz_class(quotient * t);
  }

  return {
    old_s, // a
    old_t, // b
    old_r  // g
  };       // s.t. ax + by = g
}

// Lifts a polynomial ∈ Z[x]/(x^(n/2) +1) to Z[x]/(x^n +1)
//
// See first term of line {11, 12} of algorithm 6, in Falcon specification
// https://falcon-sign.info/falcon.pdf
//
// Adapts
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/ntrugen.py#L78-L87
template<const size_t N>
static inline std::array<mpz_class, 2 * N>
lift(const std::array<mpz_class, N>& poly)
  requires((N >= 1) && (N & (N - 1)) == 0)
{
  std::array<mpz_class, N * 2> res{};
  for (size_t i = 0; i < N; i++) {
    res[2 * i] = poly[i];
  }

  return res;
}

// Galois conjugate of a polynomial f ∈ Z[x]/(x^n +1), is simply computed by
// computing f(-x), following
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/ntrugen.py#L52-L58
//
// This function is required for second term of line {11, 12} of algorithm 6, in
// Falcon specification https://falcon-sign.info/falcon.pdf
template<const size_t N>
static inline std::array<mpz_class, N>
galois_conjugate(const std::array<mpz_class, N>& poly)
  requires((N >= 1) && (N & (N - 1)) == 0)
{
  std::array<mpz_class, N> res;
  for (size_t i = 0; i < N; i++) {
    res[i] = mpz_class{ 1 + (-2) * (i & 1ul) } * poly[i];
  }

  return res;
}

}
