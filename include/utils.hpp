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

}
