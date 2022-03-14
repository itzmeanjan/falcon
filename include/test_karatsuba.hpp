#pragma once
#include "karatsuba.hpp"

namespace test {

// Tests data-parallel karatsuba polynomial multiplication implementation, **in
// a static manner**
//
// @todo Improve test such that it can work for other input sizes and values.
void
karatsuba(sycl::queue& q)
{
  using namespace karatsuba;

  const size_t i_dim = 8;
  const size_t o_dim = i_dim << 1;
  const size_t wg_size = 4;

  std::vector<double> res = { 1,   4,   10,  20,  35,  56,  84, 120,
                              147, 164, 170, 164, 145, 112, 64, 0 };

  const size_t i_size = sizeof(double) * i_dim;
  const size_t o_size = sizeof(double) * o_dim;

  double* src_a = static_cast<double*>(sycl::malloc_shared(i_size, q));
  double* src_b = static_cast<double*>(sycl::malloc_shared(i_size, q));
  double* itmd = static_cast<double*>(sycl::malloc_shared(i_size, q));
  double* dst = static_cast<double*>(sycl::malloc_shared(o_size, q));

  // a = [1, 2, 3, 4, 5, 6, 7, 8]
  // b = a
  // n = len(a)
  //
  // now invoke
  // https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/ntrugen.py#L14
  //
  // res = [1, 4, 10, 20, 35, 56, 84, 120, 147, 164, 170, 164, 145, 112, 64, 0]
  for (size_t i = 0; i < i_dim; i++) {
    src_a[i] = i + 1;
    src_b[i] = i + 1;
  }

  std::vector<sycl::event> evts = multiplication(
    q, src_a, i_dim, src_b, i_dim, itmd, i_dim, dst, o_dim, wg_size, {});

  q.ext_oneapi_submit_barrier(evts).wait();

  for (size_t i = 0; i < o_dim; i++) {
    assert(res[i] == dst[i]);
  }

  sycl::free(src_a, q);
  sycl::free(src_b, q);
  sycl::free(itmd, q);
  sycl::free(dst, q);
}

}
