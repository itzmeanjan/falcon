#pragma once
#include "common.hpp"
#include "u72.hpp"

namespace samplerz {

// Upper bound on all the values of sigma
// See
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/samplerz.py#L9-L11
constexpr double MAX_SIGMA = 1.8205;
constexpr double INV_2SIGMA2 = 1.0 / (2 * (MAX_SIGMA * MAX_SIGMA));

// Bit precision of RCDT
// See
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/samplerz.py#L13-L14
constexpr uint64_t RCDT_PREC = 72;

// Reverse cumulative distribution table, taken from column three of table 3.1
// of Falcon specification https://falcon-sign.info/falcon.pdf
const std::string RCDT[19] = { "3024686241123004913666",
                               "1564742784480091954050",
                               "636254429462080897535",
                               "199560484645026482916",
                               "47667343854657281903",
                               "8595902006365044063",
                               "1163297957344668388",
                               "117656387352093658",
                               "8867391802663976",
                               "496969357462633",
                               "20680885154299",
                               "638331848991",
                               "14602316184",
                               "247426747",
                               "3104126",
                               "28824",
                               "198",
                               "1",
                               "0" };

// Constants taken from step 1 of algorithm 13 in Falcon specification
// https://falcon-sign.info/falcon.pdf
constexpr uint64_t C[13] = { 0x00000004741183A3ul, 0x00000036548CFC06ul,
                             0x0000024FDCBF140Aul, 0x0000171D939DE045ul,
                             0x0000D00CF58F6F84ul, 0x000680681CF796E3ul,
                             0x002D82D8305B0FEAul, 0x011111110E066FD0ul,
                             0x0555555555070F00ul, 0x155555555581FF00ul,
                             0x400000000002B400ul, 0x7FFFFFFFFFFF4800ul,
                             0x8000000000000000ul };

// See algorithm 12 of Falcon specification https://falcon-sign.info/falcon.pdf;
// you may also want to see
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/samplerz.py#L65-L76
const uint32_t
base_sampler()
{
  std::array<uint8_t, 9> bytes;
  random_bytes(bytes.size(), bytes.data());

  const u72::u72 u = u72::from_decimal(bytes.data());
  uint32_t z0 = 0;

  auto v = RCDT[0];

  for (size_t i = 0; i < 19; i++) {
    u72::u72 v = u72::from_decimal(RCDT[i].data(), RCDT[i].size());

    z0 += (u72::cmp(u, v) == -1); // u < v ? 1 : 0
  }

  return z0;
}

// Multiplies two 63 -bit unsigned integers and returns top 63 -bits of
// 126 -bits result
//
// Note, underlying data type is 64 -bits wide, which is why some bit
// manipulation required to get top 63 -bits of result !
static inline const uint64_t
top_63_bits(const uint64_t a, const uint64_t b)
{
  // returns MSB 64 -bits of 128 -bits result
  uint64_t high = sycl::mul_hi(a, b);
  // returns LSB 64 -bits of 128 -bits result
  uint64_t low = a * b;

  // finally keep MSB 65 -bits of 128 -bits result
  return (high << 1) | (low >> 63);
}

// Returns an integral approximation of 2 ^ 63 * ccs * exp(−x); see algorithm 13
// in Falcon specification https://falcon-sign.info/falcon.pdf
const uint64_t
approx_exp(const double x, const double ccs)
{
  uint64_t y = C[0];
  uint64_t z = static_cast<uint64_t>(x * (1ul << 63));

  for (size_t i = 1; i < 13; i++) {
    y = C[i] - top_63_bits(y, z);
  }

  z = static_cast<uint64_t>(ccs * (1ul << 63)) << 1;
  return top_63_bits(y, z);
}

}
