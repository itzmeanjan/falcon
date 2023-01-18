#pragma once
#include "samplerz.hpp"

// Generate f, g, F, G ∈ Z[x]/(φ) | fG − gF = q mod φ ( i.e. NTRU equation )
namespace ntru_gen {

// Generate a random polynomial of degree (n - 1) | n ∈ {512, 1024} and each
// coefficient is sampled from a gaussian distribution D_{Z, σ{f, g}, 0} with σ
// = 1.17 * √(q/ 8192) as described in equation 3.29 on page 34 of the Falcon
// specification https://falcon-sign.info/falcon.pdf
template<const size_t n>
static inline void
gen_poly(int32_t* const poly)
{
  constexpr double σ = 1.43300980528773;
  constexpr size_t k = 4096 / n;

  for (size_t i = 0; i < n; i++) {

    int32_t res = 0;
    for (size_t j = 0; j < k; j++) {
      if constexpr (n == 512) {
        res += samplerz::samplerz(0., σ, samplerz::FALCON512_σ_min);
      } else {
        res += samplerz::samplerz(0., σ, samplerz::FALCON1024_σ_min);
      }
    }

    poly[i] = res;
  }
}

}
