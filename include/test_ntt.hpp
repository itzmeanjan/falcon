#pragma once
#include "ntt.hpp"

namespace test {

void
ntt(sycl::queue& q, const size_t dim, const size_t wg_size)
{
  using namespace ntt;

  const size_t size = sizeof(uint32_t) * dim;

  uint32_t* src = static_cast<uint32_t*>(sycl::malloc_shared(size, q));
  uint32_t* fft_dst = static_cast<uint32_t*>(sycl::malloc_shared(size, q));
  uint32_t* ifft_dst = static_cast<uint32_t*>(sycl::malloc_shared(size, q));

  random_fill(src, dim);
  sycl::event evt0 = q.memset(fft_dst, 0, size);
  sycl::event evt1 = q.memset(ifft_dst, 0, size);

  std::vector<sycl::event> evts0 =
    cooley_tukey_ntt(q, src, fft_dst, dim, wg_size, { evt0 });
  std::vector<sycl::event> evts1 =
    cooley_tukey_intt(q,
                      fft_dst,
                      ifft_dst,
                      dim,
                      wg_size,
                      { evt0, evt1, evts0.at(evts0.size() - 1) });
  evts1.at(evts1.size() - 1).wait();

  for (size_t i = 0; i < dim; i++) {
    assert(src[i] == ifft_dst[i]);
  }

  sycl::free(src, q);
  sycl::free(fft_dst, q);
  sycl::free(ifft_dst, q);
}

}
