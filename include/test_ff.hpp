#pragma once
#include "common.hpp"
#include "ff.hpp"
#include <cassert>

namespace test {

void
ff_math(sycl::queue& q)
{
  const size_t len = ff::Q;
  const size_t size = sizeof(uint32_t) * len;

  uint32_t* in_a = static_cast<uint32_t*>(sycl::malloc_shared(size, q));
  uint32_t* in_b = static_cast<uint32_t*>(sycl::malloc_shared(size, q));
  uint32_t* out_sub = static_cast<uint32_t*>(sycl::malloc_shared(size, q));
  uint32_t* out_neg = static_cast<uint32_t*>(sycl::malloc_shared(size, q));
  uint32_t* out_mul = static_cast<uint32_t*>(sycl::malloc_shared(size, q));
  uint32_t* out_div = static_cast<uint32_t*>(sycl::malloc_shared(size, q));

  random_fill(in_a, len);
  random_fill(in_b, len);

  sycl::event evt0 = q.memset(out_sub, 0, size);
  sycl::event evt1 = q.memset(out_neg, 0, size);
  sycl::event evt2 = q.memset(out_mul, 0, size);
  sycl::event evt3 = q.memset(out_div, 0, size);
  sycl::event evt4 = q.submit([&](sycl::handler& h) {
    h.depends_on({ evt0, evt1, evt2, evt3 });

    h.single_task([=]() {
      for (size_t i = 0; i < len; i++) {
        out_sub[i] = ff::sub(in_a[i], in_b[i]);
        out_neg[i] = ff::neg(in_b[i]);

        out_mul[i] = ff::mul(in_a[i], in_b[i]);
        out_div[i] = ff::div(out_mul[i], in_a[i]);
      }
    });
  });
  evt4.wait();

  for (size_t i = 0; i < len; i++) {
    assert(ff::add(in_a[i], out_neg[i]) == out_sub[i]);
    assert(in_b[i] == out_div[i]);
  }

  sycl::free(in_a, q);
  sycl::free(in_b, q);
  sycl::free(out_sub, q);
  sycl::free(out_neg, q);
  sycl::free(out_mul, q);
  sycl::free(out_div, q);
}

}
