#pragma once
#include "ntru_gen.hpp"

namespace test {

void
sqnorm(sycl::queue& q, const size_t dim, const size_t wg_size)
{
  using events = std::vector<sycl::event>;

  const size_t itmd_size = 4096 * sizeof(int32_t);
  const size_t poly_size = dim * sizeof(int32_t);
  const size_t norm_size = sizeof(uint32_t);

  int32_t* itmd = static_cast<int32_t*>(sycl::malloc_shared(itmd_size, q));
  int32_t* poly = static_cast<int32_t*>(sycl::malloc_shared(poly_size, q));
  uint32_t* norm = static_cast<uint32_t*>(sycl::malloc_shared(norm_size, q));

  events evts0 = ntru::gen_poly(q, dim, wg_size, itmd, poly, {});
  events evts1 = ntru::sqnorm(q, poly, dim, norm, wg_size, evts0);

  q.ext_oneapi_submit_barrier(evts1).wait();

  uint32_t h_norm = 0;
  for (size_t i = 0; i < dim; i++) {
    h_norm += static_cast<uint32_t>(poly[i] * poly[i]);
  }

  assert(h_norm == norm[0]);

  sycl::free(itmd, q);
  sycl::free(poly, q);
  sycl::free(norm, q);
}

}
