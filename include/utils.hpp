#pragma once
#include "fft.hpp"

namespace utils {

// Given a polynomial where each coefficent is represented as complex number,
// but it happens to be that only real part of those complex numbers are useful,
// so this function extracts out real part from each complex coefficients, in
// data parallel fashion
sycl::event
extract_real_from_complex(sycl::queue& q,
                          const fft::cmplx* const __restrict in,
                          double* const __restrict out,
                          const size_t len, // in and out element count
                          const size_t wg_size,
                          std::vector<sycl::event> evts)
{
  assert(len % wg_size == 0);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);
    h.parallel_for(sycl::nd_range<1>{ len, wg_size }, [=](sycl::nd_item<1> it) {
      const size_t idx = it.get_global_linear_id();

      out[idx] = in[idx].real();
    });
  });
}

}
