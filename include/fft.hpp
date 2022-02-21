#pragma once
#include <CL/sycl.hpp>
#include <cassert>
#include <complex>

namespace fft {

using cmplx = std::complex<double>;

class kernelFFTPrepareComplexInput;
class kernelFFTButterfly;
class kernelFFTFinalReorder;

// $ python3
// >>> import math
// >>> math.pi
constexpr double PI = 3.141592653589793;
constexpr double Θ_FFT_512 = (-2. * PI) / 512.;
constexpr double Θ_INV_FFT_512 = -Θ_FFT_512;
constexpr double Θ_FFT_1024 = (-2. * PI) / 1024.;
constexpr double Θ_INV_FFT_1024 = -Θ_FFT_1024;

// $ python3
// >>> math.cos((-2 * math.pi) / 512), math.sin((-2 * math.pi) / 512)
constexpr cmplx ω_fft_512 = cmplx(0.9999247018391445, -0.012271538285719925);

// $ python3
// >>> math.cos((2 * math.pi) / 512), math.sin((2 * math.pi) / 512)
constexpr cmplx ω_ifft_512 = cmplx(0.9999247018391445, 0.012271538285719925);

// $ python3
// >>> math.cos((-2 * math.pi) / 1024), math.sin((-2 * math.pi) / 1024)
constexpr cmplx ω_fft_1024 = cmplx(0.9999811752826011, -0.006135884649154475);

// $ python3
// >>> math.cos((2 * math.pi) / 1024), math.sin((2 * math.pi) / 1024)
constexpr cmplx ω_ifft_1024 = cmplx(0.9999811752826011, 0.006135884649154475);

const size_t
bin_log(size_t n)
{
  size_t cnt = 0ul;

  while (n > 1ul) {
    cnt++;
    n >>= 1;
  }

  return cnt;
}

const size_t
bit_rev(const size_t v, const size_t max_bit_width)
{
  size_t v_rev = 0ul;
  for (size_t i = 0; i < max_bit_width; i++) {
    v_rev += ((v >> i) & 0b1) * (1ul << (max_bit_width - 1ul - i));
  }
  return v_rev;
}

const size_t
rev_all_bits(const size_t n)
{
  size_t rev = 0;

  for (uint8_t i = 0; i < 64; i++) {
    if ((1ul << i) & n) {
      rev |= (1ul << (63 - i));
    }
  }

  return rev;
}

const size_t
permute_index(const size_t idx, const size_t size)
{
  if (size == 1ul) {
    return 0ul;
  }

  size_t bits = sycl::ext::intel::ctz(size);
  return rev_all_bits(idx) >> (64ul - bits);
}

static constexpr bool
check_fft_domain_size(const size_t dim)
{
  return dim == 512 || dim == 1024;
}

template<size_t dim>
void
cooley_tukey_fft(sycl::queue& q,
                 const double* const __restrict src,
                 cmplx* const __restrict dst,
                 const size_t wg_size) requires(check_fft_domain_size(dim))
{
  assert((dim & (dim - 1)) == 0); // power of 2 check
  assert(dim >= wg_size);
  assert(dim % wg_size == 0); // all work groups have same # -of work items

  const size_t log2dim = bin_log(dim);

  std::vector<sycl::event> evts;
  evts.reserve(log2dim + 2);

  sycl::event evt0 = q.parallel_for<kernelFFTPrepareComplexInput>(
    sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
    [=](sycl::nd_item<1> it) {
      const size_t k = it.get_global_linear_id();

      dst[k] = cmplx(src[k], 0.);
    });

  evts.push_back(evt0);

  for (size_t i = log2dim - 1ul; i >= 0ul; i--) {
    sycl::event evt1 = q.submit([=](sycl::handler& h) {
      h.depends_on(evts.at(log2dim - (i + 1)));

      h.parallel_for<kernelFFTButterfly>(
        sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
        [=](sycl::nd_item<1> it) {
          const size_t k = it.get_global_linear_id();
          const size_t p = 1ul << i;
          const size_t q = dim >> i;

          const size_t k_rev = bit_rev(k, log2dim) % q;
          const cmplx ω = dim == 512 ? std::pow(ω_fft_512, p * k_rev)
                                     : std::pow(ω_fft_1024, p * k_rev);

          if (k < (k ^ p)) {
            const cmplx tmp_k = dst[k];
            const cmplx tmp_k_p = dst[k ^ p];
            const cmplx tmp_k_p_ω = tmp_k_p * ω;

            dst[k] = tmp_k + tmp_k_p_ω;
            dst[k ^ p] = tmp_k - tmp_k_p_ω;
          }
        });
    });

    evts.push_back(evt1);
  }

  sycl::event evt2 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts.at(log2dim));

    h.parallel_for<kernelFFTFinalReorder>(
      sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t k = it.get_global_linear_id();
        const size_t k_perm = permute_index(k, dim);

        if (k_perm > k) {
          cmplx a = dst[k];
          cmplx b = dst[k_perm];

          const cmplx tmp = a;
          a = b;
          b = tmp;

          dst[k] = a;
          dst[k_perm] = b;
        }
      });
  });

  evts.push_back(evt2);
  evt2.wait();
}

}
