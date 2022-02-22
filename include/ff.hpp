#pragma once
#include <CL/sycl.hpp>

namespace ff {

typedef struct xgcd_t
{
  const int32_t a;
  const int32_t b;
  const int32_t g;
} xgcd_t;

// Prime Field Modulus for Falcon; see
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/common.py#L4-L5
constexpr uint32_t Q = 12 * 1024 + 1;

// Primitive Element of prime field
// $ python3
// >>> import galois as gl
// >>> gf = gl.GF(12289)
// >>> gf.primitive_element
// GF(11, order=12289)
constexpr uint32_t GENERATOR = 11;

// $ python3
// >>> assert is_odd((Q - 1) >> 12)
//
// See
// https://github.com/novifinancial/winterfell/blob/86d05a2c9e6e43297db30c9822a68b9dfba439e3/math/src/field/f64/mod.rs#L196-L198
constexpr uint32_t TWO_ADICITY = 12;

// $ python3
// >>> import galois as gl
// >>> gf = gl.GF(12289)
// >>> k = (gf.order - 1) >> 12
// >>> gf.primitive_element ** k
// GF(1331, order=12289)
constexpr uint32_t TWO_ADIC_ROOT_OF_UNITY = 1331;

const uint32_t
get_root_of_unity(uint32_t n)
{
  uint32_t power = 1ul << (TWO_ADICITY - n);
  return TWO_ADIC_ROOT_OF_UNITY * power;
}

// Extended GCD algorithm for computing inverse of prime ( = Q ) field element;
// see https://aszepieniec.github.io/stark-anatomy/basic-tools
const xgcd_t
xgcd(const uint32_t x, const uint32_t y)
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

// Computes canonical form of multiplicative inverse of prime field element,
// where a ∈ F_Q; Q = field modulus; ensure 0 < a < Q
//
// Say return value of this function is b, then
//
// assert (a * b) % Q == 1
//
// See `Field` class definition in
// https://aszepieniec.github.io/stark-anatomy/basic-tools
const uint32_t
inv(const uint32_t a // operand to be inverted; must be in `0 < a < Q`
)
{
  // can't compute multiplicative inverse of 0 in prime field
  if (a == 0) {
    return 0;
  }

  xgcd_t v = xgcd(a, ff::Q);

  if (v.a < 0) {
    return ff::Q + v.a;
  }

  return v.a % ff::Q;
}

// Computes canonical form of prime field multiplication of a, b, where both of
// them belongs to [0, Q)
//
// See `Field` class definition in
// https://aszepieniec.github.io/stark-anatomy/basic-tools
static inline const uint32_t
mul(const uint32_t a, const uint32_t b)
{
  return (a * b) % ff::Q;
}

// Computes canonical form of prime field addition of a, b
//
// See `Field` class definition in
// https://aszepieniec.github.io/stark-anatomy/basic-tools
static inline const uint32_t
add(const uint32_t a, const uint32_t b)
{
  return (a + b) % ff::Q;
}

// Computes canonical form of prime field subtraction of `b` from `a`
//
// See `Field` class definition in
// https://aszepieniec.github.io/stark-anatomy/basic-tools
static inline const uint32_t
sub(const uint32_t a, const uint32_t b)
{
  return (ff::Q + a - b) % ff::Q;
}

// Computes canonical form of prime field negation of `a`
//
// See `Field` class definition in
// https://aszepieniec.github.io/stark-anatomy/basic-tools
static inline const uint32_t
neg(const uint32_t a)
{
  return (ff::Q - a) % ff::Q;
}

// Computes canonical form of prime field `a` divided by `b`, when b > 0
//
// See `Field` class definition in
// https://aszepieniec.github.io/stark-anatomy/basic-tools
static inline const uint32_t
div(const uint32_t a, const uint32_t b)
{
  // can't divide by 0 in prime field, because can't compute
  // multiplicative inverse of 0
  if (b == 0) {
    return 0;
  }

  return mul(a, inv(b));
}

// Raises field element `a` to `b` -th power; using exponentiation by squaring
// rule; see
// https://github.com/itzmeanjan/ff-gpu/blob/89c9719e5897e57e92a3989d7d8c4e120b3aa311/ff_p.cpp#L78-L101
const uint32_t
exp(const uint32_t a, const size_t b)
{
  if (b == 0) {
    return 1;
  }
  if (b == 1) {
    return a;
  }
  if (a == 0) {
    return 0;
  }

  uint32_t base = a;
  uint32_t r = b & 0b1 ? a : 1;

  // i in 1..64 - power.leading_zeros()
  for (uint8_t i = 1; i < 64 - sycl::clz(b); i++) {
    base = mul(base, base);
    if ((b >> i) & 0b1) {
      r = mul(r, base);
    }
  }

  return r % ff::Q;
}

}
