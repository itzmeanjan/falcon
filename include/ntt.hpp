#pragma once
#include "common.hpp"
#include "ff.hpp"
#include <cassert>

namespace ntt {

// Predeclaring kernel names to avoid name mangling
class kernelNTTButterfly;
class kernelNTTFinalReorder;
class kernelINTTButterfly;
class kernelINTTFinalReorder;

// See
// https://github.com/itzmeanjan/ff-gpu/blob/89c9719e5897e57e92a3989d7d8c4e120b3aa311/ntt.cpp#L10-L19
static inline const uint32_t
compute_ntt_ω(const size_t dim)
{
  return ff::get_nth_root_of_unity(bin_log(dim));
}

// See
// https://github.com/itzmeanjan/ff-gpu/blob/89c9719e5897e57e92a3989d7d8c4e120b3aa311/ntt.cpp#L21-L30
static inline const uint32_t
compute_intt_ω(const size_t dim)
{
  return ff::inv(ff::get_nth_root_of_unity(bin_log(dim)));
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
cooley_tukey_ntt(sycl::queue& q,
                 const uint32_t* const __restrict src,
                 uint32_t* const __restrict dst,
                 const size_t dim,
                 const size_t wg_size,
                 std::vector<sycl::event> dep_evts)
{
  assert((dim & (dim - 1)) == 0); // power of 2 check
  assert(dim >= wg_size);
  assert(dim % wg_size == 0); // all work groups have same # -of work items

  const size_t log2dim = bin_log(dim);
  const uint32_t ω_ntt = compute_ntt_ω(dim);

  std::vector<sycl::event> evts;
  evts.reserve(log2dim + 2);

  sycl::event evt0 = q.submit([&](sycl::handler& h) {
    h.depends_on(dep_evts);
    h.memcpy(dst, src, sizeof(uint32_t) * dim);
  });

  evts.push_back(evt0);

  for (int64_t i = log2dim - 1ul; i >= 0; i--) {
    sycl::event evt1 = q.submit([=](sycl::handler& h) {
      h.depends_on(evts.at(log2dim - (i + 1)));

      h.parallel_for<kernelNTTButterfly>(
        sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
        [=](sycl::nd_item<1> it) {
          const size_t k = it.get_global_linear_id();
          const size_t p = 1ul << i;
          const size_t q = dim >> i;

          const size_t k_rev = bit_rev(k, log2dim) % q;
          const uint32_t ω = ff::exp(ω_ntt, p * k_rev);

          if (k < (k ^ p)) {
            const uint32_t tmp_k = dst[k];
            const uint32_t tmp_k_p = dst[k ^ p];
            const uint32_t tmp_k_p_ω = ff::mul(tmp_k_p, ω);

            dst[k] = ff::add(tmp_k, tmp_k_p_ω);
            dst[k ^ p] = ff::sub(tmp_k, tmp_k_p_ω);
          }
        });
    });

    evts.push_back(evt1);
  }

  sycl::event evt2 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts.at(log2dim));

    h.parallel_for<kernelNTTFinalReorder>(
      sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t k = it.get_global_linear_id();
        const size_t k_perm = permute_index(k, dim);

        if (k_perm > k) {
          const uint32_t a = dst[k];
          const uint32_t b = dst[k_perm];

          dst[k] = b;
          dst[k_perm] = a;
        }
      });
  });

  evts.push_back(evt2);

  return evts;
}

std::vector<sycl::event>
cooley_tukey_intt(sycl::queue& q,
                  const uint32_t* const __restrict src,
                  uint32_t* const __restrict dst,
                  const size_t dim,
                  const size_t wg_size,
                  std::vector<sycl::event> dep_evts)
{
  assert((dim & (dim - 1)) == 0); // power of 2 check
  assert(dim >= wg_size);
  assert(dim % wg_size == 0); // all work groups have same # -of work items

  const size_t log2dim = bin_log(dim);
  const uint32_t ω_intt = compute_intt_ω(dim);
  const uint32_t inv_dim = ff::inv(dim);

  std::vector<sycl::event> evts;
  evts.reserve(log2dim + 2);

  sycl::event evt0 = q.submit([&](sycl::handler& h) {
    h.depends_on(dep_evts);
    h.memcpy(dst, src, sizeof(uint32_t) * dim);
  });

  evts.push_back(evt0);

  for (int64_t i = log2dim - 1ul; i >= 0; i--) {
    sycl::event evt1 = q.submit([=](sycl::handler& h) {
      h.depends_on(evts.at(log2dim - (i + 1)));

      h.parallel_for<kernelINTTButterfly>(
        sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
        [=](sycl::nd_item<1> it) {
          const size_t k = it.get_global_linear_id();
          const size_t p = 1ul << i;
          const size_t q = dim >> i;

          const size_t k_rev = bit_rev(k, log2dim) % q;
          const uint32_t ω = ff::exp(ω_intt, p * k_rev);

          if (k < (k ^ p)) {
            const uint32_t tmp_k = dst[k];
            const uint32_t tmp_k_p = dst[k ^ p];
            const uint32_t tmp_k_p_ω = ff::mul(tmp_k_p, ω);

            dst[k] = ff::add(tmp_k, tmp_k_p_ω);
            dst[k ^ p] = ff::sub(tmp_k, tmp_k_p_ω);
          }
        });
    });

    evts.push_back(evt1);
  }

  sycl::event evt2 = q.submit([&](sycl::handler& h) {
    h.depends_on(evts.at(log2dim));

    h.parallel_for<kernelINTTFinalReorder>(
      sycl::nd_range<1>{ sycl::range<1>{ dim }, sycl::range<1>{ wg_size } },
      [=](sycl::nd_item<1> it) {
        const size_t k = it.get_global_linear_id();
        const size_t k_perm = permute_index(k, dim);

        if (k_perm == k) {
          dst[k] = ff::mul(dst[k], inv_dim);
        } else if (k_perm > k) {
          const uint32_t a = dst[k];
          const uint32_t b = dst[k_perm];

          dst[k] = ff::mul(b, inv_dim);
          dst[k_perm] = ff::mul(a, inv_dim);
        }
      });
  });

  evts.push_back(evt2);

  return evts;
}

}
