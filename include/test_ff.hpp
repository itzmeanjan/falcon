#pragma once
#include "ff.hpp"
#include <cassert>

namespace test {

void
ff_math(sycl::queue& q)
{
  const size_t len = ff::Q;
  const size_t i_size = sizeof(uint32_t) * len;
  const size_t o_size = i_size;

  uint32_t* in = static_cast<uint32_t*>(sycl::malloc_shared(i_size, q));
  uint32_t* out = static_cast<uint32_t*>(sycl::malloc_shared(o_size, q));

  for (size_t i = 0; i < len; i++) {
    in[i] = static_cast<uint32_t>(i);
  }

  sycl::event evt0 = q.memset(out, 0, o_size);
  sycl::event evt1 = q.submit([&](sycl::handler& h) {
    h.depends_on(evt0);

    h.single_task([=]() {
      for (size_t i = 0; i < len; i++) {
        out[i] = ff::inv(in[i], ff::Q);
      }
    });
  });
  evt1.wait();

  // can't compute multiplicative inverse of 0
  assert(in[0] == 0 && out[0] == 0);

  for (size_t i = 1; i < len; i++) {
    assert(ff::mul(in[i], out[i]) == 1);
  }

  sycl::free(in, q);
  sycl::free(out, q);
}

}
