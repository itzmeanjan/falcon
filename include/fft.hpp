#pragma once
#include <CL/sycl.hpp>
#include <cassert>
#include <complex>

namespace fft {

using cmplx = std::complex<double>;

// Predeclaring kernel names to avoid name mangling
class kernelFFTPrepareComplexInput;
class kernelFFTButterfly;
class kernelFFTFinalReorder;
class kernelIFFTButterfly;
class kernelIFFTFinalReorder;

// $ python3
// >>> import math
// >>> math.pi
constexpr double PI = 3.141592653589793;

// Computes cosΘ + isinΘ, where Θ = -2π/ N | N = FFT domain size
static inline const cmplx
compute_fft_ω(const size_t dim)
{
  return cmplx(std::cos((-2. * PI) / static_cast<double>(dim)),
               std::sin((-2. * PI) / static_cast<double>(dim)));
}

// Computes cosΘ + isinΘ, where Θ = 2π/ N | N = IFFT domain size
static inline const cmplx
compute_ifft_ω(const size_t dim)
{
  return cmplx(std::cos((2. * PI) / static_cast<double>(dim)),
               std::sin((2. * PI) / static_cast<double>(dim)));
}

// Computes binary logarithm of `n`, when n is power of 2
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

// Given `n`, computes `m` such that
//
// $ python3
// >>> m = int('0b' + ''.join(reversed(bin(n)[2:])), base=2)
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

std::vector<sycl::event>
cooley_tukey_fft(sycl::queue& q,
                 const double* const __restrict src,
                 cmplx* const __restrict dst,
                 const size_t dim,
                 const size_t wg_size,
                 std::vector<sycl::event> dep_evts)
{
  assert((dim & (dim - 1)) == 0); // power of 2 check
  assert(dim >= wg_size);
  assert(dim % wg_size == 0); // all work groups have same # -of work items

  const size_t log2dim = bin_log(dim);
  const cmplx ω_fft = compute_fft_ω(dim);

  std::vector<sycl::event> evts;
  evts.reserve(log2dim + 2);

  sycl::event evt0 = q.submit([&](sycl::handler& h) {
    h.depends_on(dep_evts);

    h.parallel_for<kernelFFTPrepareComplexInput>(
      sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t k = it.get_global_linear_id();

        dst[k] = cmplx(src[k], 0.);
      });
  });

  evts.push_back(evt0);

  for (int64_t i = log2dim - 1ul; i >= 0; i--) {
    sycl::event evt1 = q.submit([=](sycl::handler& h) {
      h.depends_on(evts.at(log2dim - (i + 1)));

      h.parallel_for<kernelFFTButterfly>(
        sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
        [=](sycl::nd_item<1> it) {
          const size_t k = it.get_global_linear_id();
          const size_t p = 1ul << i;
          const size_t q = dim >> i;

          const size_t k_rev = bit_rev(k, log2dim) % q;
          const cmplx ω = std::pow(ω_fft, p * k_rev);

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

  return evts;
}

std::vector<sycl::event>
cooley_tukey_ifft(sycl::queue& q,
                  const cmplx* const __restrict src,
                  cmplx* const __restrict dst,
                  const size_t dim,
                  const size_t wg_size,
                  std::vector<sycl::event> dep_evts)
{
  assert((dim & (dim - 1)) == 0); // power of 2 check
  assert(dim >= wg_size);
  assert(dim % wg_size == 0); // all work groups have same # -of work items

  const size_t log2dim = bin_log(dim);
  const cmplx ω_ifft = compute_ifft_ω(dim);
  const double inv_dim = 1. / static_cast<double>(dim);

  std::vector<sycl::event> evts;
  evts.reserve(log2dim + 2);

  sycl::event evt0 = q.submit([&](sycl::handler& h) {
    h.depends_on(dep_evts);

    h.memcpy(dst, src, sizeof(cmplx) * dim);
  });
  evts.push_back(evt0);

  for (int64_t i = log2dim - 1ul; i >= 0; i--) {
    sycl::event evt1 = q.submit([=](sycl::handler& h) {
      h.depends_on(evts.at(log2dim - (i + 1)));

      h.parallel_for<kernelIFFTButterfly>(
        sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
        [=](sycl::nd_item<1> it) {
          const size_t k = it.get_global_linear_id();
          const size_t p = 1ul << i;
          const size_t q = dim >> i;

          const size_t k_rev = bit_rev(k, log2dim) % q;
          const cmplx ω = std::pow(ω_ifft, p * k_rev);

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

    h.parallel_for<kernelIFFTFinalReorder>(
      sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t k = it.get_global_linear_id();
        const size_t k_perm = permute_index(k, dim);

        if (k_perm == k) {
          dst[k] *= inv_dim;
        } else if (k_perm > k) {
          const cmplx a = dst[k];
          const cmplx b = dst[k_perm];

          dst[k] = b * inv_dim;
          dst[k_perm] = a * inv_dim;
        }
      });
  });

  evts.push_back(evt2);

  return evts;
}

}
