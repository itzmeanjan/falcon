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

// See algorithm 12 of Falcon specification https://falcon-sign.info/falcon.pdf;
// you may also want to see
// https://github.com/tprest/falcon.py/blob/88d01ede1d7fa74a8392116bc5149dee57af93f2/samplerz.py#L65-L76
uint32_t
basesampler()
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

}
