#pragma once
#include <array>
#include <gmp.h>
#include <gmpxx.h>

// Karatsuba Multiplication of two Polynomials
namespace karatsuba {

// Given two polynomials of degree N-1 ( s.t. N is power of 2 and N >= 1), this
// routine multiplies them using Karatsuba algorithm, following
// https://github.com/tprest/falcon.py/blob/88d01ed/ntrugen.py#L14-L39
// computing resulting polynomial with degree 2*N - 1
//
// Note, polynomial coefficients can be big integers - this routine depends on
// C++ interface of GNU MP
// https://gmplib.org/manual/C_002b_002b-Interface-Integers
template<const size_t N>
static inline std::array<mpz_class, 2 * N>
karatsuba(const std::array<mpz_class, N>& polya,
          const std::array<mpz_class, N>& polyb)
{
  static_assert((N & (N - 1)) == 0,
                "Degree of Polynomial + 1 must be 2^i | i >=0");

  if constexpr (N == 1) {
    return { mpz_class(polya[0] * polyb[0]), mpz_class(0) };
  } else {
    constexpr size_t Nby2 = N / 2;

    std::array<mpz_class, Nby2> polya0;
    std::array<mpz_class, Nby2> polya1;
    std::array<mpz_class, Nby2> polyb0;
    std::array<mpz_class, Nby2> polyb1;
    std::array<mpz_class, Nby2> polyax;
    std::array<mpz_class, Nby2> polybx;

    for (size_t i = 0; i < Nby2; i++) {
      polya0[i] = polya[i];
      polya1[i] = polya[Nby2 + i];

      polyb0[i] = polyb[i];
      polyb1[i] = polyb[Nby2 + i];

      polyax[i] = polya[i] + polya[Nby2 + i];
      polybx[i] = polyb[i] + polyb[Nby2 + i];
    }

    const std::array<mpz_class, N> polya0b0 = karatsuba<Nby2>(polya0, polyb0);
    const std::array<mpz_class, N> polya1b1 = karatsuba<Nby2>(polya1, polyb1);
    std::array<mpz_class, N> polyaxbx = karatsuba<Nby2>(polyax, polybx);

    for (size_t i = 0; i < N; i++) {
      polyaxbx[i] = polyaxbx[i] - mpz_class(polya0b0[i] + polya1b1[i]);
    }

    std::array<mpz_class, 2 * N> polyab{};
    for (size_t i = 0; i < N; i++) {
      polyab[i] = polyab[i] + polya0b0[i];
      polyab[N + i] = polyab[N + i] + polya1b1[i];
      polyab[Nby2 + i] = polyab[Nby2 + i] + polyaxbx[i];
    }

    return polyab;
  }
}

// Given two polynomials of degree N-1 ( s.t. N is power of 2 and N>=1 ), this
// routine first multiplies them using Karatsuba algorithm and then reduces it
// modulo  (x ** N + 1), following
// https://github.com/tprest/falcon.py/blob/88d01ed/ntrugen.py#L42-L49
template<const size_t N>
static inline std::array<mpz_class, N>
karamul(const std::array<mpz_class, N>& polya,
        const std::array<mpz_class, N>& polyb)
{
  const std::array<mpz_class, 2 * N> polyab = karatsuba(polya, polyb);

  std::array<mpz_class, N> res{};
  for (size_t i = 0; i < N; i++) {
    res[i] = polyab[i] - polyab[N + i];
  }

  return res;
}

}
