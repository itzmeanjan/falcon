#pragma once
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>

// Prime field arithmetic over Z_q, for Falcon PQC s.t. q = 3 * (2 ^ 12) + 1
namespace ff {

// Falcon Prime Field Modulus
constexpr uint16_t Q = 3 * (1 << 12) + 1;

// Precomputed Barrett Reduction Constant
//
// Note,
//
// k = ceil(log2(Q)) = 14
// r = floor((1 << 2k) / Q) = 21843
//
// See https://www.nayuki.io/page/barrett-reduction-algorithm for more.
constexpr uint16_t R = 21843;

// Extended GCD algorithm for computing multiplicative inverse over
// prime field Z_q
//
// Taken from
// https://github.com/itzmeanjan/kyber/blob/3cd41a5/include/ff.hpp#L49-L82
// Extended GCD algorithm for computing inverse of prime ( = Q ) field element
//
// Taken from
// https://github.com/itzmeanjan/falcon/blob/45b0593215c3f2ec550860128299b123885b3a42/include/ff.hpp#L40-L67
static inline constexpr std::array<int16_t, 3>
xgcd(const uint16_t x, const uint16_t y)
{
  int16_t old_r = static_cast<int16_t>(x), r = static_cast<int16_t>(y);
  int16_t old_s = 1, s = 0;
  int16_t old_t = 0, t = 1;

  while (r != 0) {
    int16_t quotient = old_r / r;
    int16_t tmp = 0;

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

  return {
    old_s, // a
    old_t, // b
    old_r  // g
  };       // s.t. `ax + by = g`
}

// Falcon Prime Field element e ∈ [0, Q), with arithmetic operations defined
// & implemented over Z_q.
struct ff_t
{
  uint16_t v = 0u;

  // Construct field element, holding canonical value _v % Q
  inline constexpr ff_t(const uint16_t _v = 0) { v = _v % Q; }

  // Construct field element, holding canonical value 0
  inline constexpr ff_t zero() { return ff_t{ 0 }; }

  // Construct field element, holding canonical value 1
  inline constexpr ff_t one() { return ff_t{ 1 }; }

  // Addition over prime field Z_q
  constexpr ff_t operator+(const ff_t& rhs) const
  {
    const uint16_t t0 = this->v + rhs.v;
    const bool flg = t0 >= Q;
    const uint16_t t1 = t0 - flg * Q;

    return ff_t{ t1 };
  }

  // Subtraction over prime field Z_q
  constexpr ff_t operator-(const ff_t& rhs) const
  {
    const ff_t t0 = -rhs;
    return *this + t0;
  }

  // Negation over prime field Z_q
  constexpr ff_t operator-() const
  {
    const uint16_t tmp = Q - this->v;
    return ff_t{ tmp };
  }

  // Multiplication over prime field Z_q
  //
  // Note, after multiplying two 14 -bit numbers resulting 28 -bit number is
  // reduced to Z_q using Barrett reduction algorithm, which avoid division by
  // any value which is not power of 2 | q = 12289.
  //
  // See https://www.nayuki.io/page/barrett-reduction-algorithm for Barrett
  // reduction algorithm
  inline constexpr ff_t operator*(const ff_t& rhs) const
  {
    const uint32_t t0 = static_cast<uint32_t>(this->v);
    const uint32_t t1 = static_cast<uint32_t>(rhs.v);
    const uint32_t t2 = t0 * t1;

    const uint64_t t3 = static_cast<uint64_t>(t2) * static_cast<uint64_t>(R);
    const uint32_t t4 = static_cast<uint32_t>(t3 >> 28);
    const uint32_t t5 = t4 * static_cast<uint32_t>(Q);
    const uint16_t t6 = static_cast<uint16_t>(t2 - t5);

    const bool flg = t6 >= Q;
    const uint16_t t7 = t6 - flg * Q;

    return ff_t{ t7 };
  }

  // Multiplicative inverse over prime field Z_q
  //
  // Say input is a & return value of this function is b, then
  //
  // assert (a * b) % q == 1
  //
  // When input a = 0, multiplicative inverse can't be computed, hence return
  // value is 0. Look out for this sitation, because this function won't raise
  // exception.
  //
  // Taken from
  // https://github.com/itzmeanjan/kyber/blob/3cd41a5/include/ff.hpp#L190-L216
  constexpr ff_t inv() const
  {
    const bool flg0 = this->v == 0;
    const uint16_t t0 = this->v + flg0 * 1;

    const auto res = xgcd(t0, Q);

    const bool flg1 = res[0] < 0;
    const uint16_t t1 = static_cast<uint16_t>(flg1 * Q + res[0]);

    const bool flg2 = t1 >= Q;
    const uint16_t t2 = t1 - flg2 * Q;
    const uint16_t t3 = t2 - flg0 * 1;

    return ff_t{ t3 };
  }

  // Division over prime field Z_q
  constexpr ff_t operator/(const ff_t& rhs) const
  {
    return (*this) * rhs.inv();
  }

  // Raises field element to N -th power, using exponentiation by repeated
  // squaring rule
  //
  // Taken from
  // https://github.com/itzmeanjan/kyber/blob/3cd41a5/include/ff.hpp#L224-L246
  constexpr ff_t operator^(const size_t n) const
  {
    ff_t base = *this;

    const ff_t br[]{ ff_t{ 1 }, base };
    ff_t res = br[n & 0b1ul];

    const size_t zeros = std::countl_zero(n);
    const size_t till = 64ul - zeros;

    for (size_t i = 1; i < till; i++) {
      base = base * base;

      const ff_t br[]{ ff_t{ 1 }, base };
      res = res * br[(n >> i) & 0b1ul];
    }

    return res;
  }
};

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
  for (uint8_t i = 1; i < 64 - std::countl_one(b); i++) {
    base = mul(base, base);
    if ((b >> i) & 0b1) {
      r = mul(r, base);
    }
  }

  return r % ff::Q;
}

// Computes root of unity of order `2 ^ n`, such that n > 0 && n <= TWO_ADICITY
//
// See
// https://github.com/novifinancial/winterfell/blob/86d05a2c9e6e43297db30c9822a68b9dfba439e3/math/src/field/traits.rs#L220-L233
static inline const uint32_t
get_nth_root_of_unity(const uint32_t n)
{
  return exp(TWO_ADIC_ROOT_OF_UNITY, 1ul << (TWO_ADICITY - n));
}

}
