#pragma once
#include "ff.hpp"
#include "samplerz.hpp"
#include <chrono>
#include <oneapi/dpl/random>

namespace ntru {

// See step 1 of algorithm 5 in Falcon specification
// https://falcon-sign.info/falcon.pdf
//
// 1.17 * sqrt((double)ff::Q / (double)(4096 << 1))
constexpr double SIGMA = 1.43300980528773;

// Generates polynomial of degree (dim - 1), with pseudo random coefficients
// sampled from discrete Gaussian distribution D_{Z, 0, SIGMA}
//
// For on-device pseudo random number generation, I'm using Intel oneAPI DPL;
// follow
// https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-library-guide/top/random-number-generator.html
//
// See step 1-6 of algorithm 5 in Falcon specification
// https://falcon-sign.info/falcon.pdf
std::vector<sycl::event>
gen_poly(sycl::queue& q,
         const size_t dim,     // == {512, 1024}
         const size_t wg_size, // all work-groups effectively of same size
         int32_t* const itmd,  // sizeof(int32_t) * 4096
         int32_t* const poly,  // sizeof(int32_t) * dim
         std::vector<sycl::event> evts)
{
  assert((dim & (dim - 1)) == 0);
  assert(dim < 4096);
  assert(dim % wg_size == 0);
  assert(4096 % wg_size == 0);

  // seed for on-device random number generator
  auto now = std::chrono::system_clock::now();
  const uint64_t seed = std::chrono::duration_cast<std::chrono::microseconds>(
                          now.time_since_epoch())
                          .count();

  sycl::event evt0 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts);
    h.parallel_for(
      sycl::nd_range<1>{ 4096, wg_size }, [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();

        // seed (pseudo) random number generator
        oneapi::dpl::minstd_rand eng{ seed, idx };
        oneapi::dpl::uniform_int_distribution<uint8_t> dis;

        // each SYCL work-item uses its own random number generator,
        // seeded with unique offset values
        itmd[idx] = samplerz::samplerz(0., SIGMA, SIGMA - 0.001, eng, dis);
      });
  });

  const size_t k = 4096 / dim;

  sycl::event evt1 = q.submit([&](sycl::handler& h) {
    h.depends_on(evt0);
    h.parallel_for(sycl::nd_range<1>{ dim, wg_size }, [=](sycl::nd_item<1> it) {
      const size_t idx = it.get_global_linear_id();

      int32_t sum = 0;
      for (size_t j = 0; j < k; j++) {
        sum += itmd[idx * k + j];
      }

      poly[idx] = sum;
    });
  });

  return { evt0, evt1 };
}

// Computes square euclidean norm of a polynomial using atomic operations
//
// See
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/common.py#L39-L45
std::vector<sycl::event>
sqnorm(sycl::queue& q,
       const int32_t* const __restrict poly,
       const size_t len_poly,
       uint32_t* const norm_poly,
       const size_t wg_size,
       std::vector<sycl::event> evts)
{
  assert(len_poly % wg_size == 0);

  sycl::event evt0 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts);
    h.memset(norm_poly, 0, sizeof(uint32_t));
  });

  sycl::event evt1 = q.submit([&](sycl::handler& h) {
    h.depends_on(evt0);
    h.parallel_for(
      sycl::nd_range<1>{ len_poly, wg_size }, [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();
        const int32_t coeff = poly[idx];

        sycl::ext::oneapi::atomic_ref<
          uint32_t,
          sycl::memory_order_relaxed,
          sycl::memory_scope_device,
          sycl::access::address_space::ext_intel_global_device_space>
          norm_ref{ norm_poly[0] };

        norm_ref.fetch_add(static_cast<uint32_t>(coeff * coeff));
      });
  });

  return { evt0, evt1 };
}

}
