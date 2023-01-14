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
  inline constexpr u72_t from_be_bytes(const std::array<uint8_t, 9> bytes)
  {
    const uint8_t hi = bytes[0];
    uint64_t lo = 0ul;

    for (size_t i = 0; i < 8; i++) {
      lo |= static_cast<uint64_t>(bytes[1 + i]) << (7 - i) * 8;
    }

    return u72_t{ hi, lo };
  }

  // Given 9 bytes, this routine computes a 72 -bit unsigned integer s.t. these
  // bytes are interpreted in big-endian byte order.
  inline constexpr u72_t from_le_bytes(const std::array<uint8_t, 9> bytes)
  {
    const uint8_t hi = bytes[8];
    uint64_t lo = 0ul;

    for (size_t i = 0; i < 8; i++) {
      lo |= static_cast<uint64_t>(bytes[i]) << (i * 8);
    }

    return u72_t{ hi, lo };
  }
};

}
