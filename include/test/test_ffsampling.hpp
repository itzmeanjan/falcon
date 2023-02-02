#pragma once
#include "ffsampling.hpp"
#include "keygen.hpp"
#include <cassert>

// Test functional correctness of Falcon PQC suite implementation
namespace test_falcon {

// Check whether we can successfully generate two polynomials (s1, s2) each of
// degree-N s.t. they satisfy the equation s1 + s2 * h = c ( mod q ), given
// matrix B  = [[g, -f], [G, -F]] ( in its FFT representation ), falcon tree T (
// in its FFT representation ) and falcon public key vector h ( in its
// coefficient form ), which are generated using keygen function ( see
// keygen.hpp file ).
//
// This function ensures that our implementation of ffSampling algorithm is
// correct and it works as expected by attempting to **partially** implement
// Falcon signing algorithm ( algo 10 in specification ).
template<const size_t N>
void
test_ff_sampling(
  const double σ,    // Standard deviation ( see table 3.3 of specification )
  const double σ_min // See table 3.3 of specification
  )
  requires((N == 512) || (N == 1024))
{
  // 2^k * (1 + k) -many complex numbers required for storing Falcon tree of
  // height k | k = log2(N)
  constexpr size_t ft_len = (1ul << log2<N>()) * (log2<N>() + 1);
  constexpr fft::cmplx q{ ff::Q };

  auto B = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N * 4));
  auto T = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * ft_len));
  auto h = static_cast<ff::ff_t*>(std::malloc(sizeof(ff::ff_t) * N));
  auto h_fft = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto c = static_cast<ff::ff_t*>(std::malloc(sizeof(ff::ff_t) * N));
  auto c_fft = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto t0 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto t1 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto z0 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto z1 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto tz0 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto tz1 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto s0 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto s1 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto s0_ntt = static_cast<ff::ff_t*>(std::malloc(sizeof(ff::ff_t) * N));
  auto s1_ntt = static_cast<ff::ff_t*>(std::malloc(sizeof(ff::ff_t) * N));
  auto tmp0 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto tmp1 = static_cast<ff::ff_t*>(std::malloc(sizeof(ff::ff_t) * N));

  // Falcon key generation, computes
  //
  // - Matrix B = [[g, -f], [G, -F]] ( FFT form )
  // - Falcon tree T ( FFT form )
  // - Falcon public key h ( Coeff Form )
  keygen::keygen<N>(B, T, h, σ);

  // Emulate line 2 of algorithm 10 of Falcon specification
  for (size_t i = 0; i < N; i++) {
    c[i] = ff::ff_t::random();
  }
  for (size_t i = 0; i < N; i++) {
    c_fft[i] = fft::cmplx{ static_cast<double>(c[i].v) };
  }

  // compute t = (t0, t1) | see line 3 of algo 10 in the specification
  fft::fft<log2<N>()>(c_fft);
  polynomial::mul<log2<N>()>(c_fft, B + 3 * N, t0);
  polynomial::mul<log2<N>()>(c_fft, B + N, t1);

  for (size_t i = 0; i < N; i++) {
    t0[i] /= q;
    t1[i] = -(t1[i] / q);
  }

  // ffSampling i.e. compute z = (z0, z1), same as line 6 of algo 10
  ffsampling::ff_sampling<N, 0, log2<N>()>(t0, t1, T, σ_min, z0, z1);

  // compute tz = (tz0, tz1) = (t0 - z0, t1 - z1)
  polynomial::sub<log2<N>()>(t0, z0, tz0);
  polynomial::sub<log2<N>()>(t1, z1, tz1);

  // compute s = (s0, s1) = tz * B | tz is 1x2 and B = 2x2 ( of dimension )
  polynomial::mul<log2<N>()>(tz0, B, s0);
  polynomial::mul<log2<N>()>(tz1, B + 2 * N, tmp0);
  polynomial::add_to<log2<N>()>(s0, tmp0);
  fft::ifft<log2<N>()>(s0);

  polynomial::mul<log2<N>()>(tz0, B + N, s1);
  polynomial::mul<log2<N>()>(tz1, B + 3 * N, tmp0);
  polynomial::add_to<log2<N>()>(s1, tmp0);
  fft::ifft<log2<N>()>(s1);

  // Coefficients of s0, s1 ∈ [-6145, 6143], moving them to ∈ [0, 12289)
  for (size_t i = 0; i < N; i++) {
    const auto v0 = static_cast<int32_t>(std::round(s0[i].real()));
    const auto v1 = static_cast<int32_t>(std::round(s1[i].real()));

    s0_ntt[i].v = static_cast<uint16_t>((v0 < 0) * 12289 + v0);
    s1_ntt[i].v = static_cast<uint16_t>((v1 < 0) * 12289 + v1);
  }

  ntt::ntt<log2<N>()>(s0_ntt);
  ntt::ntt<log2<N>()>(s1_ntt);
  ntt::ntt<log2<N>()>(h);

  polynomial::mul<log2<N>()>(s1_ntt, h, tmp1); // tmp = s1 * h
  polynomial::add_to<log2<N>()>(tmp1, s0_ntt); // tmp = s0 + tmp

  ntt::intt<log2<N>()>(tmp1);

  // ensure that s0 + s1 * h = c ( mod q ) satisfies !
  bool match = true;
  for (size_t i = 0; i < N; i++) {
    match &= tmp1[i] == c[i];
  }

  std::free(B);
  std::free(T);
  std::free(h);
  std::free(h_fft);
  std::free(c);
  std::free(c_fft);
  std::free(t0);
  std::free(t1);
  std::free(z0);
  std::free(z1);
  std::free(tz0);
  std::free(tz1);
  std::free(s0);
  std::free(s1);
  std::free(s0_ntt);
  std::free(s1_ntt);
  std::free(tmp0);
  std::free(tmp1);

  assert(match);
}

}
