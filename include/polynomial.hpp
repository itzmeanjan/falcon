#pragma once
#include "fft.hpp"

namespace polynomial {

class kernelAddCoeffPolynomials;
class kernelNegatePolynomial;
class kernelSubCoeffPolynomials;
class kernelMulCoeffPolynomials;
class kernelDivCoeffPolynomials;

// Adds two coefficient representation polynomials; see
// https://github.com/tprest/falcon.py/blob/3a6fe63db658ff88fbdcb52c0899fe171b86370a/fft.py#L96-L100
sycl::event
add(sycl::queue& q,
    const double* const __restrict src_a,
    const size_t len_a,
    const double* const __restrict src_b,
    const size_t len_b,
    double* const __restrict dst,
    const size_t len_dst,
    const size_t wg_size,
    std::vector<sycl::event> evts)
{
  assert(len_a == len_b);
  assert(len_b == len_dst);

  sycl::event evt = q.submit([&](sycl::handler& h) {
    h.depends_on(evts);

    h.parallel_for<kernelAddCoeffPolynomials>(
      sycl::nd_range<1>{ sycl::range<1>{ len_dst }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();

        dst[idx] = src_a[idx] + src_b[idx];
      });
  });

  return evt;
}

// Negates coefficient representation polynomial; see
// https://github.com/tprest/falcon.py/blob/3a6fe63db658ff88fbdcb52c0899fe171b86370a/fft.py#L103-L106
sycl::event
neg(sycl::queue& q,
    const double* const __restrict src,
    const size_t len_src,
    double* const __restrict dst,
    const size_t len_dst,
    const size_t wg_size,
    std::vector<sycl::event> evts)
{
  assert(len_src == len_dst);

  sycl::event evt = q.submit([&](sycl::handler& h) {
    h.depends_on(evts);

    h.parallel_for<kernelNegatePolynomial>(
      sycl::nd_range<1>{ sycl::range<1>{ len_dst }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();

        dst[idx] = -src[idx];
      });
  });

  return evt;
}

// Subtracts coefficient representation polynomials; see
// https://github.com/tprest/falcon.py/blob/3a6fe63db658ff88fbdcb52c0899fe171b86370a/fft.py#L109-L111
sycl::event
sub(sycl::queue& q,
    const double* const __restrict src_a,
    const size_t len_a,
    const double* const __restrict src_b,
    const size_t len_b,
    double* const __restrict dst,
    const size_t len_dst,
    const size_t wg_size,
    std::vector<sycl::event> evts)
{
  assert(len_a == len_dst);
  assert(len_b == len_dst);

  sycl::event evt = q.submit([&](sycl::handler& h) {
    h.depends_on(evts);

    h.parallel_for<kernelSubCoeffPolynomials>(
      sycl::nd_range<1>{ sycl::range<1>{ len_dst }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();

        dst[idx] = src_a[idx] - src_b[idx];
      });
  });

  return evt;
}

// Multiplies two coefficient representation polynomials; see
// https://github.com/tprest/falcon.py/blob/3a6fe63db658ff88fbdcb52c0899fe171b86370a/fft.py#L114-L116
std::vector<sycl::event>
mul(sycl::queue& q,
    const double* const __restrict src_a_coeff, // input polynomial 1
    fft::cmplx* const __restrict src_a_fft,
    const size_t len_a,
    const double* const __restrict src_b_coeff, // input polynomial 2
    fft::cmplx* const __restrict src_b_fft,
    const size_t len_b,
    fft::cmplx* const __restrict dst_coeff, // result is here
    fft::cmplx* const __restrict dst_fft,
    const size_t len_dst,
    const size_t wg_size,
    std::vector<sycl::event> evts)
{
  assert(len_a == len_dst);
  assert(len_b == len_dst);

  std::vector<sycl::event> evts0 =
    fft::cooley_tukey_fft(q, src_a_coeff, src_a_fft, len_a, wg_size, evts);
  std::vector<sycl::event> evts1 =
    fft::cooley_tukey_fft(q, src_b_coeff, src_b_fft, len_b, wg_size, evts);

  sycl::event evt = q.submit([&](sycl::handler& h) {
    h.depends_on({ evts0.at(evts0.size() - 1), evts1.at(evts1.size() - 1) });

    h.parallel_for<kernelMulCoeffPolynomials>(
      sycl::nd_range<1>{ sycl::range<1>{ len_dst }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();

        dst_fft[idx] = src_a_fft[idx] * src_b_fft[idx];
      });
  });

  std::vector<sycl::event> evts2 =
    fft::cooley_tukey_ifft(q, dst_fft, dst_coeff, len_dst, wg_size, { evt });
  return evts2;
}

// Performs division of two coefficient representation polynomials; see
// https://github.com/tprest/falcon.py/blob/3a6fe63db658ff88fbdcb52c0899fe171b86370a/fft.py#L119-L121
std::vector<sycl::event>
div(sycl::queue& q,
    const double* const __restrict src_a_coeff, // input polynomial 1
    fft::cmplx* const __restrict src_a_fft,
    const size_t len_a,
    const double* const __restrict src_b_coeff, // input polynomial 2
    fft::cmplx* const __restrict src_b_fft,
    const size_t len_b,
    fft::cmplx* const __restrict dst_coeff, // result is here
    fft::cmplx* const __restrict dst_fft,
    const size_t len_dst,
    const size_t wg_size,
    std::vector<sycl::event> evts)
{
  assert(len_a == len_dst);
  assert(len_b == len_dst);

  std::vector<sycl::event> evts0 =
    fft::cooley_tukey_fft(q, src_a_coeff, src_a_fft, len_a, wg_size, evts);
  std::vector<sycl::event> evts1 =
    fft::cooley_tukey_fft(q, src_b_coeff, src_b_fft, len_b, wg_size, evts);

  sycl::event evt = q.submit([&](sycl::handler& h) {
    h.depends_on({ evts0.at(evts0.size() - 1), evts1.at(evts1.size() - 1) });

    h.parallel_for<kernelDivCoeffPolynomials>(
      sycl::nd_range<1>{ sycl::range<1>{ len_dst }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();

        dst_fft[idx] = src_a_fft[idx] / src_b_fft[idx];
      });
  });

  std::vector<sycl::event> evts2 =
    fft::cooley_tukey_ifft(q, dst_fft, dst_coeff, len_dst, wg_size, { evt });
  return evts2;
}

}
