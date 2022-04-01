#pragma once
#include "fft.hpp"
#include "karatsuba.hpp"

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

// Given a polynomial generated using `ntru::gen_poly`, such that each randomly
// sampled coefficient is a double precision floating point number, this
// function converts each negative coefficient to it's canonical representation
// in Zq | q = 12289, using `ff:neg` function
//
// If a coefficient is >= 0., it's just casted to proper data type ( uint32_t )
// so that it can be consumed by cooley-tukey NTT compute routine !
sycl::event
to_canonical_zq_coefficients(sycl::queue& q,
                             const double* const __restrict poly_in,
                             uint32_t* const __restrict poly_out,
                             const size_t len,
                             const size_t wg_size,
                             std::vector<sycl::event> evts)
{
  assert(len % wg_size == 0);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);
    h.parallel_for(sycl::nd_range<1>{ len, wg_size }, [=](sycl::nd_item<1> it) {
      const size_t idx = it.get_global_linear_id();
      const double elm = poly_in[idx];

      if (elm < 0.) {
        // compute canonical form only when element is negative
        poly_out[idx] = ff::neg(static_cast<uint32_t>(0. - elm));
      } else {
        // otherwise just cast to NTT consumable data type
        poly_out[idx] = static_cast<uint32_t>(elm);
      }
    });
  });
}

// Given a polynomial in NTT representation, this function checks whether all
// coefficients are non-zero or not.
//
// Ensure that `nonzero_cnt` argument is initialised to `1`; if after completion
// of this offloaded compute job, it's found that all coefficients are nonzero,
// `nonzero_cnt` should hold still `1`, otherwise it'll be zeroed.
//
// This implementation uses atomic operation !
sycl::event
is_nonzero_coeff(
  sycl::queue& q,
  const uint32_t* const __restrict poly,
  const size_t len,
  const size_t wg_size,
  uint32_t* const __restrict nonzero, // must be initialised to `1` ( true )
  std::vector<sycl::event> evts)
{
  assert(len % wg_size == 0);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);
    h.parallel_for(sycl::nd_range<1>{ len, wg_size }, [=](sycl::nd_item<1> it) {
      const size_t idx = it.get_global_linear_id();
      const uint32_t elm = poly[idx];

      sycl::ext::oneapi::atomic_ref<
        uint32_t,
        sycl::memory_order_relaxed,
        sycl::memory_scope_device,
        sycl::access::address_space::ext_intel_global_device_space>
        nonzero_ref{ nonzero[0] };

      nonzero_ref.fetch_and(static_cast<uint32_t>(elm != 0u));
    });
  });
}

// See
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/ntrugen.py#L52-L58
sycl::event
galois_conjugate(
  sycl::queue& q,
  const double* const __restrict poly_src, // `poly_len` -many coefficients
  double* const __restrict poly_dst,       // `poly_len` -many coefficients
  const size_t poly_len,
  const size_t wg_size,
  std::vector<sycl::event> evts)
{
  assert(poly_len % wg_size == 0);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);
    h.parallel_for(sycl::nd_range<1>{ poly_len, wg_size },
                   [=](sycl::nd_item<1> it) {
                     const size_t idx = it.get_global_linear_id();
                     const double elm = poly_src[idx];

                     if ((idx & 0b1ul) == 1) { // odd
                       poly_dst[idx] = -1. * elm;
                     } else { // even
                       poly_dst[idx] = elm;
                     }
                   });
  });
}

// See
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/ntrugen.py#L78-L87
sycl::event
lift(sycl::queue& q,
     const double* const __restrict poly_src,
     const size_t poly_len_src,
     double* const __restrict poly_dst,
     const size_t poly_len_dst,
     const size_t wg_size,
     std::vector<sycl::event> evts)
{
  assert(poly_len_src << 1 == poly_len_dst);
  assert(poly_len_src % wg_size == 0);

  sycl::event evt0 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts);
    h.memset(poly_dst, 0, sizeof(double) * poly_len_dst);
  });

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evt0);
    h.parallel_for(
      sycl::nd_range<1>{ poly_len_src, wg_size }, [=](sycl::nd_item<1> it) {
        const size_t frm_idx = it.get_global_linear_id(); // read from
        const size_t to_idx = frm_idx << 1;               // write to

        poly_dst[to_idx] = poly_src[frm_idx];
      });
  });
}

// Given a polynomial with N -many coefficients, this routine selects only those
// coefficients which are living on even indices and places them contiguously as
// coefficients of a polynomial with (N >> 1) -many coefficients
sycl::event
select_even(sycl::queue& q,
            const double* const __restrict poly_src,
            const size_t poly_src_len, // = poly_dst_len << 1
            double* const __restrict poly_dst,
            const size_t poly_dst_len, // = poly_src_len >> 1
            const size_t wg_size,
            std::vector<sycl::event> evts)
{
  assert(poly_src_len == poly_dst_len << 1);
  assert(poly_dst_len % wg_size == 0);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);
    h.parallel_for(sycl::nd_range<1>{ poly_dst_len, wg_size },
                   [=](sycl::nd_item<1> it) {
                     const size_t to_idx = it.get_global_linear_id();
                     const size_t frm_idx = to_idx << 1; // even index

                     poly_dst[to_idx] = poly_src[frm_idx];
                   });
  });
}

// Given a polynomial with N -many coefficients, this routine selects only those
// coefficients which are living on odd indices and places them contiguously as
// coefficients of a polynomial with (N >> 1) -many coefficients
//
// Just same as `utils::select_even` routine, only selecting odd indices from
// source polynomial
sycl::event
select_odd(sycl::queue& q,
           const double* const __restrict poly_src,
           const size_t poly_src_len, // = poly_dst_len << 1
           double* const __restrict poly_dst,
           const size_t poly_dst_len, // = poly_src_len >> 1
           const size_t wg_size,
           std::vector<sycl::event> evts)
{
  assert(poly_src_len == poly_dst_len << 1);
  assert(poly_dst_len % wg_size == 0);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts);
    h.parallel_for(sycl::nd_range<1>{ poly_dst_len, wg_size },
                   [=](sycl::nd_item<1> it) {
                     const size_t to_idx = it.get_global_linear_id();
                     const size_t frm_idx = (to_idx << 1) + 1ul; // odd index
                     poly_dst[to_idx] = poly_src[frm_idx];
                   });
  });
}

// Routine used for mapping polynomial `f` ∈ Z[x] / (x^n + 1) to a smaller ring
// Z[x] / (x^(n/2) + 1), using equation 3.25 in Falcon specification
// https://falcon-sign.info/falcon.pdf
//
// Here `poly_src` is polynomial `f` in coefficient representation, which is to
// be projected to a smaller ring & `poly_dst` will hold the projected one
//
// See
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/ntrugen.py#L61-L75
sycl::event
field_norm(
  sycl::queue& q,
  const double* const __restrict poly_src, // `poly_len` -many coefficients
  double* const __restrict poly_even,      // `poly_len` -many coefficients
  double* const __restrict poly_odd,       // `poly_len` -many coefficients
  double* const __restrict poly_even_sqrd, // `poly_len` -many coefficients
  double* const __restrict poly_odd_sqrd,  // `poly_len` -many coefficients
  double* const __restrict poly_itmd_0,    // `poly_len` -many coefficients
  double* const __restrict poly_itmd_1,    // `poly_len` -many coefficients
  double* const __restrict poly_itmd_2,    // `poly_len << 1` -many coefficients
  double* const __restrict poly_itmd_3,    // `poly_len << 1` -many coefficients
  double* const __restrict poly_itmd_4,    // `poly_len << 1` -many coefficients
  double* const __restrict poly_itmd_5,    // `poly_len << 1` -many coefficients
  double* const __restrict poly_dst,       // `poly_len` -many coefficients
  const size_t poly_len,                   // power of 2
  const size_t wg_size,
  std::vector<sycl::event> evts)
{
  assert((poly_len >> 1) % wg_size == 0);
  assert((poly_len & (poly_len - 1)) == 0);

  using evt = sycl::event;
  using evnts = std::vector<evt>;

  const size_t half_len = poly_len >> 1;
  const size_t half_size = half_len * sizeof(double);

  evt evt0 =
    select_even(q, poly_src, poly_len, poly_even, half_len, wg_size, evts);

  evt evt1 =
    select_odd(q, poly_src, poly_len, poly_odd, half_len, wg_size, evts);

  evt evt2 = q.submit([&](sycl::handler& h) {
    h.depends_on(evt0);
    h.memcpy(poly_itmd_0, poly_even, half_size);
  });

  evt evt3 = q.submit([&](sycl::handler& h) {
    h.depends_on(evt1);
    h.memcpy(poly_itmd_1, poly_odd, half_size);
  });

  evnts evts0 = karatsuba::modular_multiplication(q,
                                                  poly_even,
                                                  poly_itmd_0,
                                                  poly_itmd_2,
                                                  poly_itmd_3,
                                                  poly_even_sqrd,
                                                  half_len,
                                                  wg_size,
                                                  { evt2 });

  evnts evts1 = karatsuba::modular_multiplication(q,
                                                  poly_odd,
                                                  poly_itmd_1,
                                                  poly_itmd_4,
                                                  poly_itmd_5,
                                                  poly_odd_sqrd,
                                                  half_len,
                                                  wg_size,
                                                  { evt3 });

  evt evt4 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts0);
    h.memcpy(poly_dst, poly_even_sqrd, half_size);
  });

  // so that it becomes easier setting up dependency chain
  evts1.push_back(evt4);

  return q.submit([&](sycl::handler& h) {
    h.depends_on(evts1);
    h.parallel_for(sycl::nd_range<1>{ half_len, wg_size },
                   [=](sycl::nd_item<1> it) {
                     const size_t idx = it.get_global_linear_id();

                     if (idx == 0) {
                       poly_dst[idx] += poly_odd_sqrd[half_len - 1ul];
                     } else {
                       poly_dst[idx] -= poly_odd_sqrd[idx - 1ul];
                     }
                   });
  });
}

}
