#pragma once
#include "common.hpp"
#include "polynomial.hpp"

namespace test {

double
mul(sycl::queue& q, const size_t dim, const size_t wg_size)
{
  using namespace fft;

  const size_t i_size = sizeof(double) * dim;
  const size_t o_size = sizeof(cmplx) * dim;

  double* src_f_coeff = static_cast<double*>(sycl::malloc_shared(i_size, q));
  double* src_g_coeff = static_cast<double*>(sycl::malloc_shared(i_size, q));
  double* src_h_coeff = static_cast<double*>(sycl::malloc_shared(i_size, q));
  cmplx* dst_coeff = static_cast<cmplx*>(sycl::malloc_shared(o_size, q));
  cmplx* src_f_fft = static_cast<cmplx*>(sycl::malloc_shared(o_size, q));
  cmplx* src_g_fft = static_cast<cmplx*>(sycl::malloc_shared(o_size, q));
  cmplx* src_h_fft = static_cast<cmplx*>(sycl::malloc_shared(o_size, q));
  cmplx* dst_fft = static_cast<cmplx*>(sycl::malloc_shared(o_size, q));

  random_fill(src_f_coeff, dim);
  random_fill(src_g_coeff, dim);

  sycl::event evt0 = q.memset(src_f_fft, 0, o_size);
  sycl::event evt1 = q.memset(src_g_fft, 0, o_size);
  sycl::event evt2 = q.memset(dst_fft, 0, o_size);
  sycl::event evt3 = q.memset(dst_coeff, 0, i_size);

  std::vector<sycl::event> evts0 = polynomial::mul(q,
                                                   src_f_coeff,
                                                   src_f_fft,
                                                   dim,
                                                   src_g_coeff,
                                                   src_g_fft,
                                                   dim,
                                                   dst_coeff,
                                                   dst_fft,
                                                   dim,
                                                   wg_size,
                                                   { evt0, evt1, evt2, evt3 });

  sycl::event evt4 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts0.at(evts0.size() - 1));

    h.parallel_for(
      sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();

        src_h_coeff[idx] = dst_coeff[idx].real();
      });
  });

  sycl::event evt5 = q.memset(src_h_fft, 0, o_size);
  std::vector<sycl::event> evts1 = polynomial::div(q,
                                                   src_h_coeff,
                                                   src_h_fft,
                                                   dim,
                                                   src_f_coeff,
                                                   src_f_fft,
                                                   dim,
                                                   dst_coeff,
                                                   dst_fft,
                                                   dim,
                                                   wg_size,
                                                   { evt4, evt5 });

  evts1.at(evts1.size() - 1).wait();

  double max_diff = 0.;

  for (size_t i = 0; i < dim; i++) {
    using namespace std;

    const double diff = abs(src_g_coeff[i] - round(dst_coeff[i].real()));
    max_diff = max(diff, max_diff);
  }

  sycl::free(src_f_coeff, q);
  sycl::free(src_g_coeff, q);
  sycl::free(src_h_coeff, q);
  sycl::free(src_f_fft, q);
  sycl::free(src_g_fft, q);
  sycl::free(src_h_fft, q);
  sycl::free(dst_coeff, q);
  sycl::free(dst_fft, q);

  return max_diff;
}

}
