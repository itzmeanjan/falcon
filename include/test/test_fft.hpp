#pragma once
#include "common.hpp"
#include "fft.hpp"

namespace test {

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
  sycl::event evt0 = q.memset(fft_dst, 0, o_size);
  sycl::event evt1 = q.memset(ifft_dst, 0, o_size);

  std::vector<sycl::event> evts0 =
    cooley_tukey_fft(q, src, fft_dst, dim, wg_size, { evt0 });
  std::vector<sycl::event> evts1 =
    cooley_tukey_ifft(q,
                      fft_dst,
                      ifft_dst,
                      dim,
                      wg_size,
                      { evt0, evt1, evts0.at(evts0.size() - 1) });
  evts1.at(evts1.size() - 1).wait();

  double max_diff = 0.;

  for (size_t i = 0; i < dim; i++) {
    const double diff = sycl::abs(src[i] - std::round(ifft_dst[i].real()));
    max_diff = std::max(diff, max_diff);
  }

  sycl::free(src, q);
  sycl::free(fft_dst, q);
  sycl::free(ifft_dst, q);

  return max_diff;
}

}
