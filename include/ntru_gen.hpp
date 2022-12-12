#pragma once
#include "ff.hpp"
#include "polynomial.hpp"
#include "samplerz.hpp"
#include "utils.hpp"
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
         int32_t* const __restrict itmd, // sizeof(int32_t) * 4096
         double* const __restrict poly,  // sizeof(double) * dim
         std::vector<sycl::event> evts)
{
  assert((dim & (dim - 1)) == 0);
  assert(dim < 4096);
  assert(dim % wg_size == 0);
  assert(4096 % wg_size == 0);

  using namespace std::chrono;
  using ns = nanoseconds;
  using tp = _V2::system_clock::time_point;

  // seed for on-device random number generator
  const tp now = system_clock::now();
  const uint64_t seed = duration_cast<ns>(now.time_since_epoch()).count();

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

      double sum = 0.;
      for (size_t j = 0; j < k; j++) {
        sum += static_cast<double>(itmd[idx * k + j]);
      }

      poly[idx] = sum;
    });
  });

  return { evt0, evt1 };
}

// Computes square of euclidean norm of a polynomial using atomic operations;
// see equation 3.9 of Falcon specification https://falcon-sign.info/falcon.pdf
//
// See
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/common.py#L39-L45
std::vector<sycl::event>
sqnorm(sycl::queue& q,
       const double* const __restrict poly,
       const size_t len_poly,
       double* const norm_poly,
       const size_t wg_size,
       std::vector<sycl::event> evts)
{
  assert(len_poly % wg_size == 0);

  sycl::event evt0 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts);
    h.memset(norm_poly, 0, sizeof(double));
  });

  sycl::event evt1 = q.submit([&](sycl::handler& h) {
    h.depends_on(evt0);
    h.parallel_for(
      sycl::nd_range<1>{ len_poly, wg_size }, [=](sycl::nd_item<1> it) {
        const size_t idx = it.get_global_linear_id();
        const double coeff = poly[idx];

        sycl::ext::oneapi::atomic_ref<
          double,
          sycl::memory_order_relaxed,
          sycl::memory_scope_device,
          sycl::access::address_space::ext_intel_global_device_space>
          norm_ref{ norm_poly[0] };

        norm_ref.fetch_add(coeff * coeff);
      });
  });

  return { evt0, evt1 };
}

// Computes squared Gram-Schmidt norm of NTRU matrix generated by two randomly
// generated polynomials `f` & `g`
//
// Equivalent to line 9 of algorithm 5 in Falcon specification
// https://falcon-sign.info/falcon.pdf
//
// I took some inspiration from
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/ntrugen.py#L190-L201
void
gs_norm(sycl::queue& q,
        const double* const __restrict poly_f,
        const size_t len_poly_f,
        const double* const __restrict poly_g,
        const size_t len_poly_g,
        double* const __restrict ret_val, // squared gram-schmidt norm
        const size_t wg_size,
        std::vector<sycl::event> evts)
{
  assert(len_poly_f == len_poly_g);

  using namespace fft;
  using events = std::vector<sycl::event>;

  // --- begin computing squared euclidean norm of two polynomials in
  // coefficient representation `f`, `g` ---
  //
  // See
  // https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/ntrugen.py#L196
  const size_t norm_size = sizeof(double) * 2;
  double* norm = static_cast<double*>(sycl::malloc_shared(norm_size, q));

  events evts0 = sqnorm(q, poly_f, len_poly_f, norm + 0, wg_size, evts);
  events evts1 = sqnorm(q, poly_g, len_poly_g, norm + 1, wg_size, evts);

  events evts2{ q.ext_oneapi_submit_barrier(evts0),
                q.ext_oneapi_submit_barrier(evts1) };

  q.ext_oneapi_submit_barrier(evts2).wait();

  const double sqnorm_fg = norm[0] + norm[1];
  // --- end computing squared euclidean norm ---

  const size_t poly_size = sizeof(cmplx) * len_poly_f;
  const size_t adj_size = sizeof(double) * len_poly_f;

  cmplx* poly_fs_fft = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));
  cmplx* poly_fd_fft = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));
  cmplx* poly_fd_coeff = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));

  cmplx* poly_gs_fft = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));
  cmplx* poly_gd_fft = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));
  cmplx* poly_gd_coeff = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));

  events evts3 = polynomial::adj(q,
                                 poly_f,
                                 poly_fs_fft,
                                 len_poly_f,
                                 poly_fd_coeff,
                                 poly_fd_fft,
                                 len_poly_f,
                                 wg_size,
                                 {});

  events evts4 = polynomial::adj(q,
                                 poly_g,
                                 poly_gs_fft,
                                 len_poly_g,
                                 poly_gd_coeff,
                                 poly_gd_fft,
                                 len_poly_g,
                                 wg_size,
                                 {});

  double* adj_f = static_cast<double*>(sycl::malloc_device(adj_size, q));
  double* adj_g = static_cast<double*>(sycl::malloc_device(adj_size, q));

  cmplx* adj_f_fft = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));
  cmplx* adj_g_fft = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));

  sycl::event evt0 = utils::extract_real_from_complex(
    q, poly_fd_coeff, adj_f, len_poly_f, wg_size, evts3);

  sycl::event evt1 = utils::extract_real_from_complex(
    q, poly_gd_coeff, adj_g, len_poly_f, wg_size, evts4);

  cmplx* itmd_0_fft = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));
  cmplx* itmd_0_coeff = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));

  cmplx* itmd_1_fft = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));
  cmplx* itmd_1_coeff = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));

  events evts5 = polynomial::mul(q,
                                 poly_f,
                                 poly_fs_fft,
                                 len_poly_f,
                                 adj_f,
                                 adj_f_fft,
                                 len_poly_f,
                                 itmd_0_coeff,
                                 itmd_0_fft,
                                 len_poly_f,
                                 wg_size,
                                 { evt0 });

  events evts6 = polynomial::mul(q,
                                 poly_g,
                                 poly_gs_fft,
                                 len_poly_g,
                                 adj_g,
                                 adj_g_fft,
                                 len_poly_g,
                                 itmd_1_coeff,
                                 itmd_1_fft,
                                 len_poly_g,
                                 wg_size,
                                 { evt1 });

  double* itmd_2 = static_cast<double*>(sycl::malloc_device(adj_size, q));
  double* itmd_3 = static_cast<double*>(sycl::malloc_device(adj_size, q));

  sycl::event evt2 = utils::extract_real_from_complex(
    q, itmd_0_coeff, itmd_2, len_poly_f, wg_size, evts5);

  sycl::event evt3 = utils::extract_real_from_complex(
    q, itmd_1_coeff, itmd_3, len_poly_f, wg_size, evts6);

  double* ffgg = static_cast<double*>(sycl::malloc_device(adj_size, q));

  sycl::event evt4 = polynomial::add(q,
                                     itmd_2,
                                     len_poly_f,
                                     itmd_3,
                                     len_poly_f,
                                     ffgg,
                                     len_poly_f,
                                     wg_size,
                                     { evt2, evt3 });

  cmplx* ft_coeff = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));
  cmplx* gt_coeff = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));

  cmplx* ft_fft = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));
  cmplx* gt_fft = static_cast<cmplx*>(sycl::malloc_device(poly_size, q));

  events evts7 = polynomial::div(q,
                                 adj_g,
                                 itmd_0_coeff,
                                 len_poly_g,
                                 ffgg,
                                 itmd_0_fft,
                                 len_poly_g,
                                 ft_coeff,
                                 ft_fft,
                                 len_poly_g,
                                 wg_size,
                                 { evt4 });

  events evts8 = polynomial::div(q,
                                 adj_f,
                                 itmd_1_coeff,
                                 len_poly_f,
                                 ffgg,
                                 itmd_1_fft,
                                 len_poly_f,
                                 gt_coeff,
                                 gt_fft,
                                 len_poly_f,
                                 wg_size,
                                 { evt4 });

  sycl::event evt5 = utils::extract_real_from_complex(
    q, ft_coeff, itmd_2, len_poly_f, wg_size, evts7);

  sycl::event evt6 = utils::extract_real_from_complex(
    q, gt_coeff, itmd_3, len_poly_f, wg_size, evts8);

  events evts9 = sqnorm(q, itmd_2, len_poly_f, norm + 0, wg_size, { evt5 });
  events evts10 = sqnorm(q, itmd_3, len_poly_f, norm + 1, wg_size, { evt6 });

  events evts11{ q.ext_oneapi_submit_barrier(evts9),
                 q.ext_oneapi_submit_barrier(evts10) };

  q.ext_oneapi_submit_barrier(evts11).wait();

  {
    using namespace ff;

    const double sqnorm_FG = static_cast<double>(Q * Q) * (norm[0] + norm[1]);
    *ret_val = std::max(sqnorm_fg, sqnorm_FG);
  }

  sycl::free(norm, q);

  sycl::free(poly_fs_fft, q);
  sycl::free(poly_fd_fft, q);
  sycl::free(poly_fd_coeff, q);

  sycl::free(poly_gs_fft, q);
  sycl::free(poly_gd_fft, q);
  sycl::free(poly_gd_coeff, q);

  sycl::free(adj_f, q);
  sycl::free(adj_g, q);

  sycl::free(adj_f_fft, q);
  sycl::free(adj_g_fft, q);

  sycl::free(itmd_0_fft, q);
  sycl::free(itmd_0_coeff, q);
  sycl::free(itmd_1_fft, q);
  sycl::free(itmd_1_coeff, q);

  sycl::free(itmd_2, q);
  sycl::free(itmd_3, q);

  sycl::free(ffgg, q);

  sycl::free(ft_coeff, q);
  sycl::free(gt_coeff, q);
  sycl::free(ft_fft, q);
  sycl::free(gt_fft, q);
}

}
