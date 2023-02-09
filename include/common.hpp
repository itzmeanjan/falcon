#pragma once
#include "ff.hpp"
#include <iomanip>
#include <random>
#include <sstream>
#include <string_view>
#include <type_traits>

// Converts a byte array into hex string; see
// https://stackoverflow.com/a/14051107
inline const std::string
to_hex(const uint8_t* const bytes, const size_t len)
{
  std::stringstream ss;
  ss << std::hex;

  for (size_t i = 0; i < len; i++) {
    ss << std::setw(2) << std::setfill('0') << static_cast<uint32_t>(bytes[i]);
  }
  return ss.str();
}

// Given a hex encoded string, this routine computes a byte array of length
// `hex_string.length() / 2` -bytes; see https://stackoverflow.com/a/30606613
inline void
to_byte_array(const std::string& hex_string, uint8_t* const __restrict bytes)
{
  const size_t slen = hex_string.length();
  const size_t blen = slen / 2;

  for (size_t i = 0; i < blen; i++) {
    const size_t off = i * 2;

    const auto t0 = hex_string.substr(off, 2);
    const auto t1 = std::strtol(t0.c_str(), nullptr, 16);
    const auto t2 = static_cast<uint8_t>(t1);

    bytes[i] = t2;
  }
}

// Compile-time compute binary logarithm of N s.t. N is power of 2 and N >= 1
template<const size_t N>
static inline constexpr size_t
log2()
  requires((N >= 1) && (N & (N - 1)) == 0)
{
  return std::bit_width(N) - 1;
}
