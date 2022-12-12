#pragma once
#include "common.hpp"
#include "ff.hpp"
#include <cassert>

namespace test {

inline void
ff_math()
{
  const size_t len = ff::Q;
  const size_t size = sizeof(uint32_t) * len;

  uint32_t* in_a = static_cast<uint32_t*>(std::malloc(size));
  uint32_t* in_b = static_cast<uint32_t*>(std::malloc(size));
  uint32_t* out_sub = static_cast<uint32_t*>(std::malloc(size));
  uint32_t* out_neg = static_cast<uint32_t*>(std::malloc(size));
  uint32_t* out_mul = static_cast<uint32_t*>(std::malloc(size));
  uint32_t* out_div = static_cast<uint32_t*>(std::malloc(size));

  random_fill<uint32_t>(in_a, len);
  random_fill<uint32_t>(in_b, len);

  for (size_t i = 0; i < len; i++) {
    out_sub[i] = ff::sub(in_a[i], in_b[i]);
    out_neg[i] = ff::neg(in_b[i]);

    out_mul[i] = ff::mul(in_a[i], in_b[i]);
    out_div[i] = ff::div(out_mul[i], in_a[i]);
  }

  for (size_t i = 0; i < len; i++) {
    assert(ff::add(in_a[i], out_neg[i]) == out_sub[i]);
    assert(in_b[i] == out_div[i]);
  }

  std::free(in_a);
  std::free(in_b);
  std::free(out_sub);
  std::free(out_neg);
  std::free(out_mul);
  std::free(out_div);
}

}
