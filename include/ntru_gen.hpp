#pragma once
#include "ff.hpp"
#include "samplerz.hpp"

namespace ntru {

// See step 1 of algorithm 5 in Falcon specification
// https://falcon-sign.info/falcon.pdf
//
// 1.17 * sqrt((double)ff::Q / (double)(4096 << 1))
constexpr double SIGMA = 1.43300980528773;

void
gen_poly(const size_t dim,    // == {512, 1024}
         int32_t* const itmd, // sizeof(int32_t) * 4096
         int32_t* const poly  // sizeof(int32_t) * dim
)
{
  assert((dim & (dim - 1)) == 0);
  assert(dim < 4096);

#pragma unroll 16
  for (size_t i = 0; i < 4096; i++) {
    itmd[i] = samplerz::samplerz(0., SIGMA, SIGMA - 0.001);
  }

  const size_t k = 4096 / dim;

#pragma unroll 16
  for (size_t i = 0; i < dim; i++) {
    int32_t sum = 0;
    for (size_t j = 0; j < k; j++) {
      sum += itmd[i * k + j];
    }

    poly[i] = sum;
  }
}

}
