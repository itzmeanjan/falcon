#include "fft.hpp"
#include "polynomial.hpp"
#include <cstring>
#include <gtest/gtest.h>

// Ensure functional correctness of (i)FFT implementation, using polynomial
// multiplication and division in FFT form, over C
//
// Test is adapted from
// https://github.com/tprest/falcon.py/blob/88d01ed/test.py#L46-L59
template<const size_t lgn>
static void
test_fft()
{
  constexpr size_t n = 1ul << lgn;

  auto* poly_a = static_cast<fft::cmplx*>(std::malloc(n * sizeof(fft::cmplx)));
  auto* poly_b = static_cast<fft::cmplx*>(std::malloc(n * sizeof(fft::cmplx)));
  auto* poly_d = static_cast<fft::cmplx*>(std::malloc(n * sizeof(fft::cmplx)));

  auto* fft_a = static_cast<fft::cmplx*>(std::malloc(n * sizeof(fft::cmplx)));
  auto* fft_b = static_cast<fft::cmplx*>(std::malloc(n * sizeof(fft::cmplx)));
  auto* fft_c = static_cast<fft::cmplx*>(std::malloc(n * sizeof(fft::cmplx)));
  auto* fft_d = static_cast<fft::cmplx*>(std::malloc(n * sizeof(fft::cmplx)));

  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<int> dis{ -3, 4 };

  for (size_t i = 0; i < n; i++) {
    poly_a[i] = fft::cmplx{ static_cast<double>(dis(gen)) };
    poly_b[i] = fft::cmplx{ static_cast<double>(dis(gen)) };
  }

  std::memcpy(fft_a, poly_a, n * sizeof(fft::cmplx));
  std::memcpy(fft_b, poly_b, n * sizeof(fft::cmplx));

  fft::fft<lgn>(fft_a);
  fft::fft<lgn>(fft_b);

  polynomial::mul<lgn>(fft_a, fft_b, fft_c); // c = a * b
  polynomial::div<lgn>(fft_c, fft_b, fft_d); // d = c / b

  std::memcpy(poly_d, fft_d, n * sizeof(fft::cmplx));
  fft::ifft<lgn>(poly_d);

  for (size_t i = 0; i < n; i++) {
    poly_d[i] = fft::cmplx{ std::round(poly_d[i].real()) };
  }

  bool flg = false;
  for (size_t i = 0; i < n; i++) {
    flg |= (poly_d[i] != poly_a[i]);
  }

  std::free(poly_a);
  std::free(poly_b);
  std::free(poly_d);
  std::free(fft_a);
  std::free(fft_b);
  std::free(fft_c);
  std::free(fft_d);

  EXPECT_FALSE(flg);
}

TEST(Falcon, PolynomialArithmeticInFFTDomain)
{
  test_fft<ntt::FALCON512_LOG2N>();
  test_fft<ntt::FALCON1024_LOG2N>();
}

// Splits a polynomial f into two polynomials f0, f1 s.t. all these polynomials
// are in their coefficient representation.
//
// This routine is an implementation of equation 3.20, described in section 3.6,
// on page 28 of the Falcon specification https://falcon-sign.info/falcon.pdf
template<const size_t lgn>
static inline void
split(const fft::cmplx* const __restrict f,
      fft::cmplx* const __restrict f0,
      fft::cmplx* const __restrict f1)
{
  constexpr size_t n = 1ul << lgn;
  constexpr size_t hn = n >> 1;

  for (size_t i = 0; i < hn; i++) {
    f0[i] = f[2 * i];
    f1[i] = f[2 * i + 1];
  }
}

// Merges two polynomials f0, f1 into a single one f s.t. all these polynomials
// are in their coefficient representation.
//
// This routine is an implementation of equation 3.22, described in section 3.6,
// on page 28 of the Falcon specification https://falcon-sign.info/falcon.pdf
template<const size_t lgn>
static inline void
merge(const fft::cmplx* const __restrict f0,
      const fft::cmplx* const __restrict f1,
      fft::cmplx* const __restrict f)
{
  constexpr size_t n = 1ul << lgn;
  constexpr size_t hn = n >> 1;

  for (size_t i = 0; i < hn; i++) {
    f[2 * i + 0] = f0[i];
    f[2 * i + 1] = f1[i];
  }
}

// Ensure that splitting and merging of polynomials in their FFT representation
// is correctly implemented, following figure 3.2, describing relationship
// between split, merge, split_fft, merge_fft, FFT and iFFT routines, on page 30
// of the Falcon specification https://falcon-sign.info/falcon.pdf
template<const size_t lgn>
static void
test_fft_split_merge()
{
  constexpr bool flg = lgn >= 2 && lgn <= 10;
  static_assert(flg, "Splits polynomials of length 2^[2..10]");

  constexpr size_t n = 1ul << lgn;
  constexpr size_t hn = n >> 1;
  constexpr size_t elen = sizeof(fft::cmplx);

  auto* poly_f = static_cast<fft::cmplx*>(std::malloc(n * elen));
  auto* poly_f0 = static_cast<fft::cmplx*>(std::malloc(hn * elen));
  auto* poly_f1 = static_cast<fft::cmplx*>(std::malloc(hn * elen));

  auto* fft_f = static_cast<fft::cmplx*>(std::malloc(n * elen));
  auto* fft_f0 = static_cast<fft::cmplx*>(std::malloc(hn * elen));
  auto* fft_f1 = static_cast<fft::cmplx*>(std::malloc(hn * elen));

  auto* ifft_f = static_cast<fft::cmplx*>(std::malloc(n * elen));
  auto* ifft_f0 = static_cast<fft::cmplx*>(std::malloc(hn * elen));
  auto* ifft_f1 = static_cast<fft::cmplx*>(std::malloc(hn * elen));

  std::random_device rd;
  std::mt19937_64 gen(rd());
  std::uniform_int_distribution<int> dis{ -3, 4 };

  for (size_t i = 0; i < n; i++) {
    poly_f[i] = fft::cmplx{ static_cast<double>(dis(gen)) };
  }

  split<lgn>(poly_f, poly_f0, poly_f1); // f -> f_0, f_1

  std::memcpy(fft_f, poly_f, n * elen);

  fft::fft<lgn>(fft_f);                       // f --(FFT)--> f^
  fft::split_fft<lgn>(fft_f, fft_f0, fft_f1); // f^ -> f^_0, f^_1

  std::memcpy(ifft_f0, fft_f0, hn * elen);
  std::memcpy(ifft_f1, fft_f1, hn * elen);

  fft::ifft<lgn - 1>(ifft_f0); // f^_0 --(iFFT)--> f_0
  fft::ifft<lgn - 1>(ifft_f1); // f^_1 --(iFFT)--> f_1

  fft::merge_fft<lgn>(fft_f0, fft_f1, ifft_f); // f^_0, f^_1 -> f^
  fft::ifft<lgn>(ifft_f);                      // f^ --(iFFT)--> f

  for (size_t i = 0; i < hn; i++) {
    ifft_f0[i] = fft::cmplx{ std::round(ifft_f0[i].real()) };
    ifft_f1[i] = fft::cmplx{ std::round(ifft_f1[i].real()) };
  }

  for (size_t i = 0; i < n; i++) {
    ifft_f[i] = fft::cmplx{ std::round(ifft_f[i].real()) };
  }

  bool flg0 = false;
  bool flg1 = false;

  for (size_t i = 0; i < hn; i++) {
    flg0 |= (ifft_f0[i] != poly_f0[i]);
    flg1 |= (ifft_f1[i] != poly_f1[i]);
  }

  bool flg2 = false;

  for (size_t i = 0; i < n; i++) {
    flg2 |= (ifft_f[i] != poly_f[i]);
  }

  std::free(poly_f);
  std::free(poly_f0);
  std::free(poly_f1);
  std::free(fft_f);
  std::free(fft_f0);
  std::free(fft_f1);
  std::free(ifft_f);
  std::free(ifft_f0);
  std::free(ifft_f1);

  EXPECT_FALSE(flg0);
  EXPECT_FALSE(flg1);
  EXPECT_FALSE(flg2);
}

TEST(Falcon, PolynomialSplitAndMergeInFFTDomain)
{
  test_fft_split_merge<2>();
  test_fft_split_merge<3>();
  test_fft_split_merge<4>();
  test_fft_split_merge<5>();
  test_fft_split_merge<6>();
  test_fft_split_merge<7>();
  test_fft_split_merge<8>();
  test_fft_split_merge<9>();
  test_fft_split_merge<10>();
}
