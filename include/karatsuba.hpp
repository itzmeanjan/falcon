#pragma once
#include <CL/sycl.hpp>
#include <cassert>

namespace karatsuba {

class kernelKaratsubaMultiplicationPhase0;
class kernelKaratsubaMultiplicationPhase1;

// Data-parallel Karatsuba polynomial multiplication, inspired from
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/ntrugen.py#L14-L39
// ( recursive ) and https://eprint.iacr.org/2006/224.pdf ( see section 3.2 for
// iterative implementation )
std::vector<sycl::event>
multiplication(sycl::queue& q,
               const double* const __restrict src_a,
               const size_t len_a,
               const double* const __restrict src_b,
               const size_t len_b,
               double* const __restrict itmd,
               const size_t len_itmd,
               double* const __restrict dst,
               const size_t len_dst,
               const size_t wg_size,
               std::vector<sycl::event> evts)
{
  assert(len_a == len_b);
  assert(len_b == len_itmd);
  assert((len_itmd << 1) == len_dst);
  assert(len_itmd % wg_size == 0);        // all workgroups are of same size
  assert((len_dst & (len_dst - 1)) == 0); // power of 2 many coefficients

  sycl::event evt0 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts);

    h.parallel_for<kernelKaratsubaMultiplicationPhase0>(
      sycl::nd_range<1>{ len_itmd, wg_size }, [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();

        itmd[idx] = src_a[idx] * src_b[idx];
      });
  });

  sycl::event evt1 = q.memset(dst, 0, sizeof(double) * len_dst);

  sycl::event evt2 = q.submit([&](sycl::handler& h) {
    h.depends_on({ evt0, evt1 });

    h.parallel_for<kernelKaratsubaMultiplicationPhase1>(
      sycl::nd_range<1>{ len_dst, wg_size }, [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();

        if (idx == 0) {
          dst[idx] = itmd[idx];
        } else if (idx == len_dst - 2) {
          dst[idx] = itmd[len_itmd - 1];
        } else {
          double d_st = 0, d_s_t = 0;

          for (size_t s = 0; s < idx; s++) {
            size_t t = idx - s;

            if ((len_itmd > t) && (t > s)) {
              d_st += ((src_a[s] + src_a[t]) * (src_b[s] + src_b[t]));
              d_s_t += (itmd[s] + itmd[t]);
            }
          }

          if ((idx & 0b1) == 0) {
            // even index
            dst[idx] = d_st - d_s_t + itmd[idx >> 1];
          } else {
            // odd index
            dst[idx] = d_st - d_s_t;
          }
        }
      });
  });

  return { evt0, evt1, evt2 };
}

}
