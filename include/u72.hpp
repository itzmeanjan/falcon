#pragma once
#include <CL/sycl.hpp>

namespace u72 {

// 72 -bit unsigned integer required for gausian sampling
typedef std::array<uint8_t, 9> u72;

static uint32_t
add_small(u72* x, uint32_t d)
{
  for (size_t i = 0; i < (*x).size(); i++) {
    uint32_t w;

    w = (*x)[i] + d;
    (*x)[i] = w & 0xff;
    d = w >> 8;
  }

  return d; // carry
}

// Given a 72 -bit unsigned integer as string ( say "12345678" ), this function
// parses it as 9 bytes
static u72
from_decimal(const char* str, const size_t len)
{
  u72 x;
  x.fill(0);

  for (size_t l = 0; l < len; l++) {
    uint32_t cc = 0;

    for (size_t i = 0; i < x.size(); i++) {
      uint32_t w;

      w = (x[i] * 10) + cc;
      x[i] = w & 0xff;
      cc = w >> 8;
    }

    add_small(&x, str[l] - '0'); // '0' = 48 ( ASCII )
  }

  return x;
}

// Given a byte array of length 9 ( read 72 randomly sampled bits ), this
// function simply copies byte array content to `u72` backing array
static u72
from_decimal(const uint8_t* bytes)
{
  u72 x;

  for (size_t l = 0; l < x.size(); l++) {
    x[l] = bytes[l];
  }

  return x;
}

// Compares two 72 -bit unsigned integers and returns
//
// -1, if x < y
//  0, if x == y
//  1, if x > y
static int8_t
cmp(const u72 x, const u72 y)
{
  for (int64_t i = x.size() - 1; i >= 0; i--) {
    if (x[i] > y[i]) {
      return 1;
    } else if (x[i] < y[i]) {
      return -1;
    }
  }

  return 0;
}

}
