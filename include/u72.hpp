#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

// Sampler over the Integers requires 72 -bit precision unsigned integer
// arithmetic and comparsion operations
namespace u72 {

// 72 -bit unsigned integer type with arithmetic ( +, - ) and comparison ( < )
// operator defined, built on top of underlying 64 -bit unsigned integers
struct u72_t
{
  uint64_t hi = 0ul;
  uint64_t lo = 0ul;

  inline constexpr u72_t(const uint64_t hi = 0ul, const uint64_t lo = 0ul)
  {
    this->hi = hi;
    this->lo = lo;
  }

  static inline constexpr u72_t zero() { return { 0ul, 0ul }; }

  // Given two 72 -bit unsigned integers, this routine adds them, computing
  // resulting 72 -bit unsigned integer
  inline constexpr u72_t operator+(const u72_t& rhs) const
  {
    const uint64_t lo = this->lo + rhs.lo;
    const bool flg = this->lo > UINT64_MAX - rhs.lo;
    const uint64_t hi = (this->hi + rhs.hi + flg * 1ul) & 0xfful;

    return { hi, lo };
  }

  // Given two 72 -bit unsigned integers, this routine subtracts one from
  // another, computing resulting 72 -bit unsigned integer
  inline constexpr u72_t operator-(const u72_t& rhs) const
  {
    const uint64_t lo = this->lo - rhs.lo;
    const bool flg = this->lo < rhs.lo;
    const uint64_t hi = (this->hi - (rhs.hi + 1ul * flg)) & 0xfful;

    return { hi, lo };
  }

  // Given a 72 -bit unsigned integer, this routine negates that element ( in ),
  // computing 72 -bit result ( out ) s.t.
  //
  // out = (2^72 - in) % 2^72 i.e. result gets wrapper around at boundary 2^72
  inline constexpr u72_t operator-() const { return u72_t::zero() - *this; }

  // Given two 72 -bit unsigned integers, this routine performs comparison s.t.
  // it returns boolean truth value if and only if lhs > rhs
  inline constexpr bool operator>(const u72_t& rhs) const
  {
    const bool flg0 = this->hi > rhs.hi;
    const bool flg1 = this->hi == rhs.hi;
    const bool flg2 = this->hi < rhs.hi;
    const bool flg3 = this->lo > rhs.lo;

    const bool flg = flg0 | (flg1 & flg3) | (!flg2 & !flg1);
    return flg;
  }

  // Given two 72 -bit unsigned integers, this routine performs comparison s.t.
  // it returns boolean truth value if and only if lhs < rhs
  inline constexpr bool operator<(const u72_t& rhs) const
  {
    return rhs > *this;
  }

  // Given 9 bytes, this routine computes a 72 -bit unsigned integer s.t. these
  // bytes are interpreted in little-endian byte order.
  static inline constexpr u72_t from_be_bytes(std::array<uint8_t, 9>&& bytes)
  {
    const uint64_t hi = static_cast<uint64_t>(bytes[0]);
    uint64_t lo = 0ul;

    for (size_t i = 0; i < 8; i++) {
      lo |= static_cast<uint64_t>(bytes[1 + i]) << (7 - i) * 8;
    }

    return { hi, lo };
  }

  // Given an unsigned 72 -bit integer, this routine interprets its bytes in
  // big-endian order and returns a byte array of length 9.
  inline constexpr std::array<uint8_t, 9> to_be_bytes() const
  {
    std::array<uint8_t, 9> res{};

    res[0] = static_cast<uint8_t>(this->hi);

    for (size_t i = 0; i < 8; i++) {
      res[1 + i] = static_cast<uint8_t>(this->lo >> (7 - i) * 8);
    }

    return res;
  }

  // Given 9 bytes, this routine computes a 72 -bit unsigned integer s.t. these
  // bytes are interpreted in big-endian byte order.
  static inline constexpr u72_t from_le_bytes(std::array<uint8_t, 9>&& bytes)
  {
    const uint64_t hi = static_cast<uint64_t>(bytes[8]);
    uint64_t lo = 0ul;

    for (size_t i = 0; i < 8; i++) {
      lo |= static_cast<uint64_t>(bytes[i]) << (i * 8);
    }

    return { hi, lo };
  }

  // Given an unsigned 72 -bit integer, this routine interprets its bytes in
  // little-endian order and returns a byte array of length 9.
  inline constexpr std::array<uint8_t, 9> to_le_bytes() const
  {
    std::array<uint8_t, 9> res{};

    for (size_t i = 0; i < 8; i++) {
      res[i] = static_cast<uint8_t>(this->lo >> i * 8);
    }
    res[8] = static_cast<uint8_t>(this->hi);

    return res;
  }
};

}
