#pragma once
#include "karatsuba.hpp"

namespace test {

// Tests data-parallel karatsuba polynomial modular multiplication
// implementation, **in a static manner**
//
// @todo Improve test such that it can work for other generic input sizes and
// values.
void
karatsuba(sycl::queue& q)
{
  constexpr size_t dim = 8;
  constexpr size_t itmd_dim = dim << 1;
  constexpr size_t wg_size = 4;

  std::array<double, dim> res = { -146, -160, -160, -144, -110, -56, 20, 120 };

  constexpr size_t size = sizeof(double) * dim;
  constexpr size_t itmd_size = sizeof(double) * itmd_dim;

  double* poly_a = static_cast<double*>(sycl::malloc_shared(size, q));
  double* poly_b = static_cast<double*>(sycl::malloc_shared(size, q));
  double* itmd_a = static_cast<double*>(sycl::malloc_shared(size, q));
  double* itmd_b = static_cast<double*>(sycl::malloc_shared(itmd_size, q));
  double* poly_dst = static_cast<double*>(sycl::malloc_shared(size, q));

  // a = [1, 2, 3, 4, 5, 6, 7, 8]
  // b = a
  // n = len(a)
  //
  // now invoke
  // https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/ntrugen.py#L42-L49
  //
  // res = [-146, -160, -160, -144, -110, -56, 20, 120]
  for (size_t i = 0; i < dim; i++) {
    poly_a[i] = i + 1;
    poly_b[i] = i + 1;
  }

  std::vector<sycl::event> evts = karatsuba::modular_multiplication(
    q, poly_a, poly_b, itmd_a, itmd_b, poly_dst, dim, wg_size, {});

  q.ext_oneapi_submit_barrier(evts).wait();

  for (size_t i = 0; i < dim; i++) {
    assert(res[i] == poly_dst[i]);
  }

  sycl::free(poly_a, q);
  sycl::free(poly_b, q);
  sycl::free(itmd_a, q);
  sycl::free(itmd_b, q);
  sycl::free(poly_dst, q);
}

}
