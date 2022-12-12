#pragma once
#include "ntru_gen.hpp"

namespace test {

// Tests correctness of function which computes square of norm of vector, using
// equation 3.9 of Falcon specification https://falcon-sign.info/falcon.pdf
void
sqnorm(sycl::queue& q, const size_t dim, const size_t wg_size)
{
  using events = std::vector<sycl::event>;

  const size_t itmd_size = 4096 * sizeof(int32_t);
  const size_t poly_size = dim * sizeof(double);
  const size_t norm_size = sizeof(double);

  int32_t* itmd = static_cast<int32_t*>(sycl::malloc_shared(itmd_size, q));
  double* poly = static_cast<double*>(sycl::malloc_shared(poly_size, q));
  double* norm = static_cast<double*>(sycl::malloc_shared(norm_size, q));

  events evts0 = ntru::gen_poly(q, dim, wg_size, itmd, poly, {});
  events evts1 = ntru::sqnorm(q, poly, dim, norm, wg_size, evts0);

  q.ext_oneapi_submit_barrier(evts1).wait();

  double h_norm = 0;
  for (size_t i = 0; i < dim; i++) {
    h_norm += (poly[i] * poly[i]);
  }

  assert(h_norm == norm[0]);

  sycl::free(itmd, q);
  sycl::free(poly, q);
  sycl::free(norm, q);
}

// Tests correctness of function which computes squared Gram-Schmidt norm of
// NTRU matrix generated from two polynomials `f`, `g`
//
// Equivalent to line 1-6 and 9-11 of algorithm 5 in Falcon specification
// https://falcon-sign.info/falcon.pdf
double
gs_norm(sycl::queue& q,
        const size_t dim,
        const size_t wg_size,
        size_t* const itr_cnt)
{
  using events = std::vector<sycl::event>;

  const size_t itmd_size = sizeof(int32_t) * 4096;
  const size_t poly_size = sizeof(double) * dim;
  const size_t ret_size = sizeof(double);

  int32_t* itmd_f = static_cast<int32_t*>(sycl::malloc_shared(itmd_size, q));
  int32_t* itmd_g = static_cast<int32_t*>(sycl::malloc_shared(itmd_size, q));
  double* poly_f = static_cast<double*>(sycl::malloc_shared(poly_size, q));
  double* poly_g = static_cast<double*>(sycl::malloc_shared(poly_size, q));
  double* ret = static_cast<double*>(sycl::malloc_shared(ret_size, q));

  // see line 10 of algorithm 5 in Falcon specification
  //
  // note, that while computing polynomial norm ( see `ntru::sqnorm` ), I've not
  // computed square root part ( following
  // https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/common.py#L39-L45
  // ), which is why, threshold to be compared against is also kept
  // as (1.17 * √q) ^ 2
  const double gs_norm_threshold = (1.17 * 1.17) * (double)ff::Q;
  size_t itr = 0;
  double norm = 0;

  // see line 9-11 of algorithm 5 in Falcon specification
  do {
    events evts0 = ntru::gen_poly(q, dim, wg_size, itmd_f, poly_f, {});
    events evts1 = ntru::gen_poly(q, dim, wg_size, itmd_g, poly_g, {});

    events evts2{ q.ext_oneapi_submit_barrier(evts0),
                  q.ext_oneapi_submit_barrier(evts1) };

    ntru::gs_norm(q, poly_f, dim, poly_g, dim, ret, wg_size, evts2);

    norm = ret[0];
    itr++;
  } while (norm > gs_norm_threshold);

  // these many iterations required before generating such `f`, `g` polynomials
  // that short signatures can be produced
  *itr_cnt = itr;

  sycl::free(itmd_f, q);
  sycl::free(itmd_g, q);
  sycl::free(poly_f, q);
  sycl::free(poly_g, q);
  sycl::free(ret, q);

  return norm;
}

}
