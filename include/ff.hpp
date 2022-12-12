#pragma once
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <random>

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

  // Equality check between two field elements ∈ Z_q
  constexpr bool operator==(const ff_t& rhs) const
  {
    return !static_cast<bool>(this->v ^ rhs.v);
  }

  // Inequality check between two field elements ∈ Z_q
  constexpr bool operator!=(const ff_t& rhs) const { return !(*this == rhs); }

  // Greater than operator applied to elements ∈ Z_q
  constexpr bool operator>(const ff_t& rhs) const { return this->v > rhs.v; }

  // Greater than equal operator applied to elements ∈ Z_q
  constexpr bool operator>=(const ff_t& rhs) const { return this->v >= rhs.v; }

  // Lesser than operator applied to elements ∈ Z_q
  constexpr bool operator<(const ff_t& rhs) const { return this->v < rhs.v; }

  // Lesser than equal operator applied to elements ∈ Z_q
  constexpr bool operator<=(const ff_t& rhs) const { return this->v <= rhs.v; }

  // Shifts operand ∈ Z_q, leftwards by l bit positions
  constexpr ff_t operator<<(const size_t l) const
  {
    return ff_t{ static_cast<uint16_t>(this->v << l) };
  }

  // Shifts operand ∈ Z_q, rightwards by l bit positions
  constexpr ff_t operator>>(const size_t l) const
  {
    return ff_t{ static_cast<uint16_t>(this->v >> l) };
  }

  // Generate a random field element ∈ Z_q
  static ff_t random()
  {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint16_t> dis{ 0, Q - 1 };

    return ff_t{ dis(gen) };
  }

  // Writes element of Z_q to output stream
  inline friend std::ostream& operator<<(std::ostream& os, const ff_t& elm);
};

std::ostream&
operator<<(std::ostream& os, const ff_t& elm)
{
  return os << "Z_q(" << elm.v << ", " << Q << ")";
}

}
