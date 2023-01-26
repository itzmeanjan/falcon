#pragma once
#include "karatsuba.hpp"
#include "polynomial.hpp"
#include "samplerz.hpp"

// Generate f, g, F, G ∈ Z[x]/(φ) | fG − gF = q mod φ ( i.e. NTRU equation )
namespace ntru_gen {

// Squared Gram-Schmidt Norm Threshold, computed following line 10 of algorithm
// 5 of Falcon specification https://falcon-sign.info/falcon.pdf
constexpr double GS_NORM_THRESHOLD = 1.17 * 1.17 * static_cast<double>(ff::Q);

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
    tmp[i].v = static_cast<uint16_t>(flg * q + poly[i]);
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
gram_schmidt_norm(const int32_t* const __restrict f,
                  const int32_t* const __restrict g)
{
  constexpr size_t N = 1ul << LOG2N;
  constexpr double q = ff::Q;
  constexpr double qxq = q * q;

  double tmp0[N];
  double tmp1[N];

  for (size_t i = 0; i < N; i++) {
    tmp0[i] = static_cast<double>(f[i]);
    tmp1[i] = static_cast<double>(g[i]);
  }

  const auto sq_norm_fg = sqrd_norm<LOG2N>(tmp0) + sqrd_norm<LOG2N>(tmp1);

  fft::cmplx f_[N];
  fft::cmplx g_[N];

  for (size_t i = 0; i < N; i++) {
    f_[i] = fft::cmplx{ tmp0[i] };
    g_[i] = fft::cmplx{ tmp1[i] };
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
// https://github.com/tprest/falcon.py/blob/88d01ed/ntrugen.py#L61-L75
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
// https://github.com/tprest/falcon.py/blob/88d01ed/ntrugen.py#L78-L87
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
// https://github.com/tprest/falcon.py/blob/88d01ed/ntrugen.py#L52-L58
//
// This function is required for second term of line {11, 12} of algorithm 6, in
// Falcon specification https://falcon-sign.info/falcon.pdf
template<const size_t N>
static inline std::array<mpz_class, N>
galois_conjugate(const std::array<mpz_class, N>& poly)
  requires((N > 1) && (N & (N - 1)) == 0)
{
  std::array<mpz_class, N> res;
  for (size_t i = 0; i < N; i++) {
    if (i & 1) {
      res[i] = -poly[i];
    } else {
      res[i] = poly[i];
    }
  }

  return res;
}

// Approximates bit length of value ∈ Z, with out considering sign bit. Note,
// this function doesn't precisely compute bit length of v, rather it rounds bit
// length to next multiple of 8.
//
// Adapted from
// https://github.com/tprest/falcon.py/blob/88d01ed/ntrugen.py#L90-L101
static inline size_t
approx_bit_len(const mpz_class& v)
{
  const mpz_class zero{ 0 };

  mpz_class v_;
  mpz_abs(v_.get_mpz_t(), v.get_mpz_t());

  size_t len = 0;
  while (v_ > zero) {
    len += 8;
    v_ = v_ >> 8;
  }

  return len;
}

// Given a polynomial of degree N ( s.t. > 1 and power of 2 ), this routine find
// minimum and maximum coefficient from that polynomial.
template<const size_t N>
static inline std::pair<mpz_class, mpz_class>
min_max(const std::array<mpz_class, N>& arr)
  requires((N > 1) && (N & (N - 1)) == 0)
{
  mpz_class min(arr[0]);
  mpz_class max(arr[0]);

  for (size_t i = 1; i < N; i++) {
    if (min > arr[i]) {
      min = arr[i];
    }
    if (max < arr[i]) {
      max = arr[i];
    }
  }

  return { min, max };
}

// Given four polynomials of degree N, this routine reduces F, G w.r.t. f, g
// using algorithm 7 of Falcon specification and returns reduced F, G.
//
// This implementation collects inspiration from
// https://github.com/tprest/falcon.py/blob/88d01ed/ntrugen.py#L104-L150
template<const size_t N>
static inline void
reduce(const std::array<mpz_class, N>& f,
       const std::array<mpz_class, N>& g,
       std::array<mpz_class, N>& F,
       std::array<mpz_class, N>& G)
  requires((N > 1) && (N & (N - 1)) == 0)
{
  const std::pair<mpz_class, mpz_class> fmm = min_max(f);
  const std::pair<mpz_class, mpz_class> gmm = min_max(g);

  const size_t blen0 = std::max(
    53ul,
    std::max(std::max(approx_bit_len(fmm.first), approx_bit_len(fmm.second)),
             std::max(approx_bit_len(gmm.first), approx_bit_len(gmm.second))));

  fft::cmplx f_adjust[N];
  fft::cmplx g_adjust[N];
  fft::cmplx f_adjoint[N];
  fft::cmplx g_adjoint[N];

  for (size_t i = 0; i < N; i++) {
    f_adjust[i] = fft::cmplx{ mpz_class(f[i] >> (blen0 - 53ul)).get_d() };
    g_adjust[i] = fft::cmplx{ mpz_class(g[i] >> (blen0 - 53ul)).get_d() };
  }

  fft::fft<log2<N>()>(f_adjust);
  fft::fft<log2<N>()>(g_adjust);

  std::memcpy(f_adjoint, f_adjust, sizeof(f_adjust));
  std::memcpy(g_adjoint, g_adjust, sizeof(g_adjust));

  fft::adj_poly<log2<N>()>(f_adjoint);
  fft::adj_poly<log2<N>()>(g_adjoint);

  while (1) {
    const std::pair<mpz_class, mpz_class> Fmm = min_max(F);
    const std::pair<mpz_class, mpz_class> Gmm = min_max(G);

    const size_t blen1 = std::max(
      53ul,
      std::max(
        std::max(approx_bit_len(Fmm.first), approx_bit_len(Fmm.second)),
        std::max(approx_bit_len(Gmm.first), approx_bit_len(Gmm.second))));

    if (blen1 < blen0) {
      break;
    }

    fft::cmplx F_adjust[N];
    fft::cmplx G_adjust[N];
    fft::cmplx F_adjoint[N];
    fft::cmplx G_adjoint[N];

    for (size_t i = 0; i < N; i++) {
      F_adjust[i] = fft::cmplx{ mpz_class(F[i] >> (blen1 - 53ul)).get_d() };
      G_adjust[i] = fft::cmplx{ mpz_class(G[i] >> (blen1 - 53ul)).get_d() };
    }

    fft::fft<log2<N>()>(F_adjust);
    fft::fft<log2<N>()>(G_adjust);

    std::memcpy(F_adjoint, F_adjust, sizeof(F_adjust));
    std::memcpy(G_adjoint, G_adjust, sizeof(G_adjust));

    fft::adj_poly<log2<N>()>(F_adjoint);
    fft::adj_poly<log2<N>()>(G_adjoint);

    fft::cmplx ff_mul[N];
    fft::cmplx gg_mul[N];
    fft::cmplx Ff_mul[N];
    fft::cmplx Gg_mul[N];

    polynomial::mul<log2<N>()>(f_adjust, f_adjoint, ff_mul);
    polynomial::mul<log2<N>()>(g_adjust, g_adjoint, gg_mul);
    polynomial::mul<log2<N>()>(F_adjust, f_adjoint, Ff_mul);
    polynomial::mul<log2<N>()>(G_adjust, g_adjoint, Gg_mul);

    fft::cmplx ffgg_add[N];
    fft::cmplx FfGg_add[N];

    polynomial::add<log2<N>()>(ff_mul, gg_mul, ffgg_add);
    polynomial::add<log2<N>()>(Ff_mul, Gg_mul, FfGg_add);

    fft::cmplx k[N];

    polynomial::div<log2<N>()>(FfGg_add, ffgg_add, k);
    fft::ifft<log2<N>()>(k);

    signed long k_rounded[N];
    for (size_t i = 0; i < N; i++) {
      k_rounded[i] = static_cast<signed long>(std::round(k[i].real()));
    }

    bool atleast_one_nonzero = false;
    for (size_t i = 0; i < N; i++) {
      atleast_one_nonzero |= k_rounded[i] != 0;
    }

    if (!atleast_one_nonzero) {
      break;
    }

    std::array<mpz_class, N> k_mpz;
    for (size_t i = 0; i < N; i++) {
      k_mpz[i] = mpz_class(k_rounded[i]);
    }

    const std::array<mpz_class, N> fk = karatsuba::karamul(f, k_mpz);
    const std::array<mpz_class, N> gk = karatsuba::karamul(g, k_mpz);

    for (size_t i = 0; i < N; i++) {
      F[i] = F[i] - mpz_class(fk[i] << (blen1 - blen0));
      G[i] = G[i] - mpz_class(gk[i] << (blen1 - blen0));
    }
  }
}

// Ad-hoc wrapper type for denoting that it's time to abort execution of NTRU
// solve algorithm
struct ntru_solve_status_t
{
private:
  uint32_t failed = 0u;

public:
  inline ntru_solve_status_t(uint32_t f = 0u) { failed = f; }
  inline bool is_solution() const { return failed == 0u; }
};

// Given two degree N polynomials f, g ∈ Z[x]/(x^N + 1), this routine attempts
// to solve NTRU equation ( see eq 3.15 of Falcon specification ), computing F,
// G ∈ Z[x]/(x^N + 1), which satisfies NTRU equation.
//
// See algorithm 6 of Falcon specification. This implementation collects
// inspiration from
// https://github.com/tprest/falcon.py/blob/88d01ed/ntrugen.py#L166-L187.
//
// Before consuming two polynomials F, G, consider checking whether it's a valid
// solution or not, using is_solution() function on returned value of type
// ntru_solve_status_t.
template<const size_t N>
static inline std::pair<
  std::pair<std::array<mpz_class, N>, std::array<mpz_class, N>>,
  ntru_solve_status_t>
ntru_solve(const std::array<mpz_class, N>& f, const std::array<mpz_class, N>& g)
  requires((N >= 1) && (N & (N - 1)) == 0)
{
  if constexpr (N == 1) {
    const auto ret = xgcd(f[0], g[0]);
    if (ret[2] != mpz_class{ 1 }) {
      return { {}, ntru_solve_status_t{ 1u } };
    } else {
      constexpr int32_t q = ff::Q;
      return { { { mpz_class(-q * ret[1]) }, { mpz_class(q * ret[0]) } },
               ntru_solve_status_t{} };
    }
  } else {
    const auto fprime = field_norm(f);
    const auto gprime = field_norm(g);

    const auto ret = ntru_solve(fprime, gprime);

    if (!ret.second.is_solution()) {
      return { {}, ret.second };
    }

    auto F = karatsuba::karamul(lift(ret.first.first), galois_conjugate(g));
    auto G = karatsuba::karamul(lift(ret.first.second), galois_conjugate(f));

    reduce(f, g, F, G);
    return { { F, G }, ntru_solve_status_t{} };
  }
}

// Given a modulus q ( = 12289 ), this routine generates four polynomials f, g,
// F, G ∈ Z[x]/(x^N + 1), solving NTRU equation ( see eq 3.15 of Falcon
// specification ). This routine is an implementation of algorithm 5 of Falcon
// specification https://falcon-sign.info/falcon.pdf
template<const size_t N>
static inline void
ntru_gen(int32_t* const __restrict f,
         int32_t* const __restrict g,
         int32_t* const __restrict F,
         int32_t* const __restrict G)
  requires((N == 512) || (N == 1024))
{
  while (1) {
    gen_poly<log2<N>()>(f);
    gen_poly<log2<N>()>(g);

    if (!is_poly_invertible<log2<N>()>(f)) {
      continue;
    }

    const double gsnorm = gram_schmidt_norm<log2<N>()>(f, g);
    if (gsnorm > GS_NORM_THRESHOLD) {
      continue;
    }

    std::array<mpz_class, N> f_;
    std::array<mpz_class, N> g_;

    for (size_t i = 0; i < N; i++) {
      f_[i] = mpz_class(f[i]);
      g_[i] = mpz_class(g[i]);
    }

    const auto ret = ntru_solve(f_, g_);
    if (!ret.second.is_solution()) {
      continue;
    }

    for (size_t i = 0; i < N; i++) {
      F[i] = static_cast<int32_t>(ret.first.first[i].get_si());
      G[i] = static_cast<int32_t>(ret.first.second[i].get_si());
    }
    break;
  }
}

}
