#pragma once
#include <CL/sycl.hpp>

namespace ff {

typedef struct xgcd_t
{
  const int32_t a;
  const int32_t b;
  const int32_t g;
} xgcd_t;

// Prime Field Modulas for Falcon; see
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/common.py#L4-L5
constexpr uint32_t Q = 12 * 1024 + 1;

// Extended GCD algorithm for computing inverse of prime ( = Q ) field element;
// see https://aszepieniec.github.io/stark-anatomy/basic-tools
xgcd_t
xgcd(uint32_t x, uint32_t y)
{
  int32_t old_r = static_cast<int32_t>(x), r = static_cast<int32_t>(y);
  int32_t old_s = 1, s = 0;
  int32_t old_t = 0, t = 1;

  while (r != 0) {
    int32_t quotient = old_r / r;
    int32_t tmp = 0;

    tmp = old_r;
    old_r = r;
    r = tmp - quotient * r;

    tmp = old_s;
    old_s = s;
    s = tmp - quotient * s;

    tmp = old_t;
    old_t = t;
    t = tmp - quotient * t;
  }

  return xgcd_t{ old_s, old_t, old_r }; // a, b, g of `ax + by = g`
}

// Computes multiplicative inverse of prime field element,
// where a ∈ F_p; p = field modulas; ensure a < p
//
// Say return value of this function is b, then
//
// assert (a * b) % p == 1
const uint32_t
inv(uint32_t a, // operand to be inverted; must be in `0 < a < p`
    uint32_t p  // prime field modulas
)
{
  if (a == 0) {
    return 0;
  }

  xgcd_t v = xgcd(a, p);

  if (v.a < 0) {
    return p + v.a;
  } else {
    return v.a % p;
  }
}

}
