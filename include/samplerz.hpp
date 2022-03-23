#pragma once
#include "common.hpp"
#include "u72.hpp"
#include <oneapi/dpl/random>

namespace samplerz {

// Upper bound on all the values of sigma
// See
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/samplerz.py#L9-L11
constexpr double MAX_SIGMA = 1.8205;
constexpr double INV_SIGMA2 = 1. / (2. * MAX_SIGMA * MAX_SIGMA);

// Bit precision of RCDT
// See
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/samplerz.py#L13-L14
constexpr uint64_t RCDT_PREC = 72;

// Reverse cumulative distribution table, taken from column three of table 3.1
// of Falcon specification https://falcon-sign.info/falcon.pdf
const char* const RCDT[19] = { "3024686241123004913666",
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

// See RCDT table; RCDT_LEN[i] = len(RCDT[i]) | i = {0, 1, 2, ... 17, 18}
const size_t RCDT_LEN[19] = { 22ul, 22ul, 21ul, 21ul, 20ul, 19ul, 19ul,
                              18ul, 16ul, 15ul, 14ul, 12ul, 11ul, 9ul,
                              7ul,  5ul,  3ul,  1ul,  1ul };

// Constants taken from step 1 of algorithm 13 in Falcon specification
// https://falcon-sign.info/falcon.pdf
constexpr uint64_t C[13] = { 0x00000004741183A3ul, 0x00000036548CFC06ul,
                             0x0000024FDCBF140Aul, 0x0000171D939DE045ul,
                             0x0000D00CF58F6F84ul, 0x000680681CF796E3ul,
                             0x002D82D8305B0FEAul, 0x011111110E066FD0ul,
                             0x0555555555070F00ul, 0x155555555581FF00ul,
                             0x400000000002B400ul, 0x7FFFFFFFFFFF4800ul,
                             0x8000000000000000ul };

// $ python3
// >>> import math
// >>> math.log(2)
constexpr double LN2 = 0.6931471805599453;
// $ python3
// >>> import math
// >>> 1/ math.log(2)
constexpr double ILN2 = 1.4426950408889634;

// See algorithm 12 of Falcon specification https://falcon-sign.info/falcon.pdf;
// you may also want to see
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/samplerz.py#L65-L76
const uint32_t
base_sampler(oneapi::dpl::minstd_rand eng,
             oneapi::dpl::uniform_int_distribution<uint8_t> dis)
{
  uint8_t bytes[9];
  for (size_t i = 0; i < 9; i++) {
    bytes[i] = dis(eng);
  }

  const u72::u72 u = u72::from_bytes(bytes);
  uint32_t z0 = 0;

  for (size_t i = 0; i < 19; i++) {
    u72::u72 v = u72::from_decimal(RCDT[i], RCDT_LEN[i]);

    z0 += (u72::cmp(u, v) == -1); // u < v ? 1 : 0
  }

  return z0;
}

// Note, this implementation is strictly for testing correctness of samplerz
// implementation; random bytes are pre-generated, taken from Falcon
// specification's table 3.2
//
// See algorithm 12 of Falcon specification https://falcon-sign.info/falcon.pdf;
// you may also want to see
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/samplerz.py#L65-L76
const uint32_t
base_sampler(uint8_t* const bytes // 9 bytes are passed
)
{
  const u72::u72 u = u72::from_bytes(bytes);
  uint32_t z0 = 0;

  for (size_t i = 0; i < 19; i++) {
    u72::u72 v = u72::from_decimal(RCDT[i], RCDT_LEN[i]);

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

// Returns a single bit = 1, with probability ≈ ccs * exp(−x)
//
// See algorithm 14 in Falcon specification https://falcon-sign.info/falcon.pdf
const bool
ber_exp(const double x,
        const double ccs,
        oneapi::dpl::minstd_rand eng,
        oneapi::dpl::uniform_int_distribution<uint8_t> dis)
{
  uint64_t s = static_cast<uint64_t>(x * ILN2);
  s = sycl::min(63ul, s);

  double r = x - static_cast<double>(s) * LN2;

  uint64_t z = (approx_exp(r, ccs) - 1) >> s;
  int64_t w = 0;

  for (int64_t i = 56; w == 0 && i > -8; i -= 8) {
    uint8_t b = dis(eng);
    w = b - ((z >> i) & 0xff);
  }

  return w < 0;
}

// Note, this implementation is strictly for testing correctness of samplerz
// implementation; random bytes are pre-generated, taken from Falcon
// specification's table 3.2
//
// Returns a single bit = 1, with probability ≈ ccs * exp(−x)
//
// See algorithm 14 in Falcon specification https://falcon-sign.info/falcon.pdf
const bool
ber_exp(const double x,
        const double ccs,
        const uint8_t* const __restrict bytes,
        size_t* const __restrict used_bytes)
{
  uint64_t s = static_cast<uint64_t>(x * ILN2);
  s = sycl::min(63ul, s);

  double r = x - static_cast<double>(s) * LN2;

  uint64_t z = (approx_exp(r, ccs) - 1) >> s;
  int64_t w = 0;

  size_t b_idx = 0;
  for (int64_t i = 56; w == 0 && i > -8; i -= 8) {
    w = bytes[b_idx++] - ((z >> i) & 0xff);
  }

  // these many bytes were consumed; read last line of page 43 of Falcon
  // specification https://falcon-sign.info/falcon.pdf; you'll understand why
  // it's necessary to keep track of it that how many bytes were used
  *used_bytes = b_idx;

  return w < 0;
}

// Sampling an integer z ∈ Z from a distribution very close to D{Z, μ, σ′}
//
// See algorithm 15 in Falcon specification https://falcon-sign.info/falcon.pdf
const int32_t
samplerz(const double mu,
         const double sigma,
         const double sigmin,
         oneapi::dpl::minstd_rand eng,
         oneapi::dpl::uniform_int_distribution<uint8_t> dis)
{
  const int32_t s = static_cast<int32_t>(sycl::floor(mu));
  const double r = mu - static_cast<double>(s);
  const double dss = 1. / (2. * sigma * sigma);
  const double ccs = sigmin / sigma;
  const uint8_t one = 0b1;

  while (true) {
    const uint32_t z0 = base_sampler(eng, dis);

    uint8_t b = dis(eng);
    b &= one; // keep only last bit

    const int32_t z = (uint32_t)b + (((uint32_t)b << 1) - 1) * z0;

    const double zr = ((double)z - r);
    const double x = (zr * zr) * dss - ((double)(z0 * z0)) * INV_SIGMA2;

    if (ber_exp(x, ccs, eng, dis)) {
      return z + s;
    }
  }
}

// Note, this implementation is strictly for testing correctness of samplerz
// implementation; random bytes are pre-generated, taken from Falcon
// specification's table 3.2
//
// Sampling an integer z ∈ Z from a distribution very close to D{Z, μ, σ′}
//
// See algorithm 15 in Falcon specification https://falcon-sign.info/falcon.pdf
const int32_t
samplerz(const double mu,
         const double sigma,
         const double sigmin,
         uint8_t* const bytes // pre-generated random bytes
)
{
  const int32_t s = static_cast<int32_t>(sycl::floor(mu));
  const double r = mu - static_cast<double>(s);
  const double dss = 1. / (2. * sigma * sigma);
  const double ccs = sigmin / sigma;

  size_t b_idx = 0;
  while (true) {
    const uint32_t z0 = base_sampler(bytes + b_idx);

    uint8_t b = bytes[b_idx + 9];
    b &= static_cast<uint8_t>(0b1); // keep only last bit

    const int32_t z = (uint32_t)b + (((uint32_t)b << 1) - 1) * z0;

    const double zr = ((double)z - r);
    const double x = (zr * zr) * dss - ((double)(z0 * z0)) * INV_SIGMA2;

    size_t used_bytes = 0;
    if (ber_exp(x, ccs, bytes + b_idx + 10, &used_bytes)) {
      return z + s;
    }

    b_idx += (10 + used_bytes);
  }
}

}
