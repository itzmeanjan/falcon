#pragma once
#include "ntru_gen.hpp"

namespace test_falcon {

// Check whether computed f, g, F, G ∈ Z[x]/(x^N + 1) actually satisfy NTRU
// equation ( see eq 3.15 of Falcon specification ) or not.
//
// This test collects some inspiration from
// https://github.com/tprest/falcon.py/blob/88d01ed/test.py#L80-L85,
// though note that for polynomial mutliplication it doesn't use Karatsuba,
// rather it performs polynomial arithmetic in frequency domain.
template<const size_t N>
static inline bool
check_ntru_eq(const int32_t* const __restrict f,
              const int32_t* const __restrict g,
              const int32_t* const __restrict F,
              const int32_t* const __restrict G)
  requires((N == 512) || (N == 1024))
{
  auto f_ = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto g_ = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto F_ = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto G_ = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto fG = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto gF = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto fGgF = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto res = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));

  for (size_t i = 0; i < N; i++) {
    f_[i] = fft::cmplx{ static_cast<double>(f[i]) };
    g_[i] = fft::cmplx{ static_cast<double>(g[i]) };
    F_[i] = fft::cmplx{ static_cast<double>(F[i]) };
    G_[i] = fft::cmplx{ static_cast<double>(G[i]) };
  }

  fft::fft<log2<N>()>(f_);
  fft::fft<log2<N>()>(g_);
  fft::fft<log2<N>()>(F_);
  fft::fft<log2<N>()>(G_);

  polynomial::mul<log2<N>()>(f_, G_, fG);
  polynomial::mul<log2<N>()>(g_, F_, gF);
  polynomial::sub<log2<N>()>(fG, gF, fGgF);

  fft::ifft<log2<N>()>(fGgF);

  bool flg = true;
  for (size_t i = 0; i < N; i++) {
    res[i] = static_cast<int32_t>(std::round(fGgF[i].real()));

    if (i == 0) {
      flg &= res[i] == static_cast<int32_t>(ff::Q);
    } else {
      flg &= res[i] == 0;
    }
  }

  std::free(f_);
  std::free(g_);
  std::free(F_);
  std::free(G_);
  std::free(fG);
  std::free(gF);
  std::free(fGgF);
  std::free(res);

  return flg;
}

}
