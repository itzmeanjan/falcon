#pragma once
#include "polynomial.hpp"
#include <cassert>
#include <cstring>

// Test functional correctness of Falcon PQC suite implementation
namespace test_falcon {

// Ensure functional correctness of (i)FFT implementation, using polynomial
// multiplication and division in FFT form, over C
//
// Test is adapted from
// https://github.com/tprest/falcon.py/blob/88d01ed/test.py#L46-L59
template<const size_t lgn>
void
test_fft()
{
  const size_t n = 1ul << lgn;

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

  assert(!flg);
}

}
