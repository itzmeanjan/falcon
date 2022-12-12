#pragma once
#include "ntru_gen.hpp"
#include "utils.hpp"
#include <cassert>

namespace test {

void
is_nonzero_coeff(sycl::queue& q, const size_t dim, const size_t wg_size)
{
  const size_t i_size = sizeof(uint32_t) * dim;
  const size_t o_size = sizeof(uint32_t) * 1;

  uint32_t* poly_0 = static_cast<uint32_t*>(sycl::malloc_shared(i_size, q));
  uint32_t* poly_1 = static_cast<uint32_t*>(sycl::malloc_shared(i_size, q));
  uint32_t* nz_acc_0 = static_cast<uint32_t*>(sycl::malloc_shared(o_size, q));
  uint32_t* nz_acc_1 = static_cast<uint32_t*>(sycl::malloc_shared(o_size, q));

  random_fill(poly_0, dim);
  random_fill(poly_1, dim);

  // explicitly putting zero, as `random_fill` won't do that
  poly_1[dim >> 1] = 0u;

  using evt = sycl::event;

  // initializing to `true` value is required !
  evt evt0 = q.single_task([=]() {
    *nz_acc_0 = 1u;
    *nz_acc_1 = 1u;
  });
  // must yield true
  evt evt1 =
    utils::is_nonzero_coeff(q, poly_0, dim, wg_size, nz_acc_0, { evt0 });
  // must yield false
  evt evt2 =
    utils::is_nonzero_coeff(q, poly_1, dim, wg_size, nz_acc_1, { evt0 });

  q.ext_oneapi_submit_barrier({ evt1, evt2 }).wait();

  bool nz_host_0 = true;
  bool nz_host_1 = true;
  for (size_t i = 0; i < dim; i++) {
    nz_host_0 &= (poly_0[i] != 0);
    nz_host_1 &= (poly_1[i] != 0);
  }

  assert((bool)nz_acc_0[0] && (bool)nz_acc_0[0] == nz_host_0);
  assert(!(bool)nz_acc_1[0] && (bool)nz_acc_1[0] == nz_host_1);

  sycl::free(poly_0, q);
  sycl::free(poly_1, q);
  sycl::free(nz_acc_0, q);
  sycl::free(nz_acc_1, q);
}

// Tests data parallel galois conjugate implementation
void
galois_conjugate(sycl::queue& q, const size_t dim, const size_t wg_size)
{
  const size_t i_size = dim * sizeof(double);
  const size_t itmd_size = 4096 * sizeof(int32_t);
  const size_t o_size = dim * sizeof(double);

  double* poly_a = static_cast<double*>(sycl::malloc_shared(i_size, q));
  int32_t* itmd = static_cast<int32_t*>(sycl::malloc_shared(itmd_size, q));
  double* poly_b = static_cast<double*>(sycl::malloc_shared(o_size, q));

  using evt = sycl::event;
  using evts = std::vector<evt>;

  evts evts0 = ntru::gen_poly(q, dim, wg_size, itmd, poly_a, {});
  evt evt0 = utils::galois_conjugate(q, poly_a, poly_b, dim, wg_size, evts0);

  evt0.wait();

  for (size_t i = 0; i < dim; i++) {
    assert(sycl::pow(-1., (double)i) * poly_a[i] == poly_b[i]);
  }

  sycl::free(poly_a, q);
  sycl::free(itmd, q);
  sycl::free(poly_b, q);
}

// Tests data parallel polynomial lift routine
void
lift(sycl::queue& q, const size_t dim, const size_t wg_size)
{
  const size_t i_size = dim * sizeof(double);
  const size_t itmd_size = 4096 * sizeof(int32_t);
  const size_t o_size = (dim << 1) * sizeof(double);

  double* poly_a = static_cast<double*>(sycl::malloc_shared(i_size, q));
  int32_t* itmd = static_cast<int32_t*>(sycl::malloc_shared(itmd_size, q));
  double* poly_b = static_cast<double*>(sycl::malloc_shared(o_size, q));

  using evt = sycl::event;
  using evts = std::vector<evt>;

  evts evts0 = ntru::gen_poly(q, dim, wg_size, itmd, poly_a, {});
  evt evt0 = utils::lift(q, poly_a, dim, poly_b, dim << 1, wg_size, evts0);

  evt0.wait();

  for (size_t i = 0; i < (dim << 1); i++) {
    if ((i & 0b1ul) == 1) { // odd
      assert(poly_b[i] == 0);
    } else { // even
      assert(poly_b[i] == poly_a[i >> 1]);
    }
  }

  sycl::free(poly_a, q);
  sycl::free(itmd, q);
  sycl::free(poly_b, q);
}

}
