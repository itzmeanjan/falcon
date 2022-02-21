#pragma once
#include "fft.hpp"
#include <random>

namespace test {

void
random_fill(double* const data, const size_t dim)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-3., 4.);

  for (size_t i = 0; i < dim; i++) {
    data[i] = dis(gen);
  }
}

double
fft(sycl::queue& q, const size_t dim, const size_t wg_size)
{
  using namespace fft;

  const size_t i_size = sizeof(double) * dim;
  const size_t o_size = sizeof(cmplx) * dim;

  double* src = static_cast<double*>(sycl::malloc_shared(i_size, q));
  cmplx* fft_dst = static_cast<cmplx*>(sycl::malloc_shared(o_size, q));
  cmplx* ifft_dst = static_cast<cmplx*>(sycl::malloc_shared(o_size, q));

  random_fill(src, dim);
  q.memset(fft_dst, 0, o_size).wait();
  q.memset(ifft_dst, 0, o_size).wait();

  cooley_tukey_fft(q, src, fft_dst, dim, wg_size);
  cooley_tukey_ifft(q, fft_dst, ifft_dst, dim, wg_size);

  double max_diff = 0.;

  for (size_t i = 0; i < dim; i++) {
    const double diff = sycl::abs(src[i] - ifft_dst[i].real());
    if (diff > max_diff) {
      max_diff = diff;
    }
  }

  sycl::free(src, q);
  sycl::free(fft_dst, q);
  sycl::free(ifft_dst, q);

  return max_diff;
}

}
