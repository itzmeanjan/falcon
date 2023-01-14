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

  inline constexpr u72_t(const uint8_t hi = 0, const uint64_t lo = 0ul)
  {
    this->hi = hi;
    this->lo = lo;
  }

  // Given 9 bytes, this routine computes a 72 -bit unsigned integer s.t. these
  // bytes are interpreted in little-endian byte order.
  static inline constexpr u72_t from_be_bytes(
    const std::array<uint8_t, 9> bytes)
  {
    const uint8_t hi = bytes[0];
    uint64_t lo = 0ul;

    for (size_t i = 0; i < 8; i++) {
      lo |= static_cast<uint64_t>(bytes[1 + i]) << (7 - i) * 8;
    }

    return u72_t{ hi, lo };
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
  static inline constexpr u72_t from_le_bytes(
    const std::array<uint8_t, 9> bytes)
  {
    const uint8_t hi = bytes[8];
    uint64_t lo = 0ul;

    for (size_t i = 0; i < 8; i++) {
      lo |= static_cast<uint64_t>(bytes[i]) << (i * 8);
    }

    return u72_t{ hi, lo };
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
