#pragma once
#include "ff.hpp"

// Falcon{512, 1024} Signature related Utility Functions
namespace falcon_utils {

// Compile-time compute public key byte length for Falcon{512, 1024}, following
// section 3.11.4 of Falcon specification
template<const size_t N>
static inline constexpr size_t
compute_pkey_len()
  requires((N == 512) || (N == 1024))
{
  constexpr size_t bw = std::bit_width(ff::Q);
  return 1ul +           // header byte
         ((N * bw) / 8); // body bytes
}

// Compile-time compute secret key byte length for Falcon{512, 1024}, following
// section 3.11.5 of Falcon specification
template<const size_t N>
static inline constexpr size_t
compute_skey_len()
  requires((N == 512) || (N == 1024))
{
  // bit width of coefficients of f, g
  constexpr size_t br[]{
    5, // N == 1024
    6  // N == 512
  };
  return 1ul +                      // header byte
         ((N * br[N == 512]) / 8) + // f as bytes
         ((N * br[N == 512]) / 8) + // g as bytes
         N;                         // F as bytes
}

// Compile-time compute Falcon{512, 1024} compressed signature byte length,
// following table 3.3 of the specification.
template<const size_t N>
static inline constexpr size_t
compute_sig_len()
  requires((N == 512) || (N == 1024))
{
  if (N == 512) {
    return 666;
  } else {
    return 1280;
  }
}

}
