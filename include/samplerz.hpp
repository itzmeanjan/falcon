#pragma once
#include "common.hpp"
#include "u72.hpp"
#include <cmath>
#include <utility>

// Sampler over the Integers
namespace samplerz {

// Scaled ( by a factor 2^72 ) Probability Distribution Table, taken from
// table 3.1 of ( on page 41 ) of Falcon specification
// https://falcon-sign.info/falcon.pdf
constexpr u72::u72_t PDT[]{ { 92ul, 579786965361551358ul },
                            { 79ul, 2650674819858381952ul },
                            { 50ul, 6151151332533475715ul },
                            { 23ul, 12418831121734727451ul },
                            { 8ul, 4319188200692788085ul },
                            { 2ul, 2177953700873134608ul },
                            { 0ul, 7432604049020375675ul },
                            { 0ul, 1045641569992574730ul },
                            { 0ul, 108788995549429682ul },
                            { 0ul, 8370422445201343ul },
                            { 0ul, 476288472308334ul },
                            { 0ul, 20042553305308ul },
                            { 0ul, 623729532807ul },
                            { 0ul, 14354889437ul },
                            { 0ul, 244322621ul },
                            { 0ul, 3075302ul },
                            { 0ul, 28626ul },
                            { 0ul, 197ul },
                            { 0ul, 1ul } };

// Compile-time computes i-th cumulative distribution | i ∈ [0, 19)
inline consteval u72::u72_t
ith_cumulative_distribution(const size_t i)
{
  auto acc = u72::u72_t::zero();

  for (size_t j = 0; j <= i; j++) {
    acc = acc + PDT[j];
  }

  return acc;
}

// Scaled ( by a factor 2^72 ) Cumulative Distribution Table, computed at
// compile-time, following formula on top of page 41 of Falcon specification
// https://falcon-sign.info/falcon.pdf
constexpr u72::u72_t CDT[]{
  ith_cumulative_distribution(0),  ith_cumulative_distribution(1),
  ith_cumulative_distribution(2),  ith_cumulative_distribution(3),
  ith_cumulative_distribution(4),  ith_cumulative_distribution(5),
  ith_cumulative_distribution(6),  ith_cumulative_distribution(7),
  ith_cumulative_distribution(8),  ith_cumulative_distribution(9),
  ith_cumulative_distribution(10), ith_cumulative_distribution(11),
  ith_cumulative_distribution(12), ith_cumulative_distribution(13),
  ith_cumulative_distribution(14), ith_cumulative_distribution(15),
  ith_cumulative_distribution(16), ith_cumulative_distribution(17),
  ith_cumulative_distribution(18),
};

// Scaled ( by a factor 2^72 ) Reverse Cumulative Distribution Table, computed
// at compile-time, following formula on top of page 41 of Falcon specification
// https://falcon-sign.info/falcon.pdf
constexpr u72::u72_t RCDT[]{ -CDT[0],  -CDT[1],  -CDT[2],  -CDT[3],  -CDT[4],
                             -CDT[5],  -CDT[6],  -CDT[7],  -CDT[8],  -CDT[9],
                             -CDT[10], -CDT[11], -CDT[12], -CDT[13], -CDT[14],
                             -CDT[15], -CDT[16], -CDT[17], -CDT[18] };

// C contains the coefficients of a polynomial that approximates e^-x
//
// More precisely, the value:
//
//  (2 ^ -63) * sum(C[12 - i] * (x ** i) for i in range(i))
//
// Should be very close to e^-x.
// This polynomial is lifted from FACCT: https://doi.org/10.1109/TC.2019.2940949
//
// These cofficients are taken from top of page 42 of Falcon specification
// https://falcon-sign.info/falcon.pdf
constexpr uint64_t C[]{ 0x00000004741183A3ul, 0x00000036548CFC06ul,
                        0x0000024FDCBF140Aul, 0x0000171D939DE045ul,
                        0x0000D00CF58F6F84ul, 0x000680681CF796E3ul,
                        0x002D82D8305B0FEAul, 0x011111110E066FD0ul,
                        0x0555555555070F00ul, 0x155555555581FF00ul,
                        0x400000000002B400ul, 0x7FFFFFFFFFFF4800ul,
                        0x8000000000000000ul };

// BaseSampler routine as defined in algorithm 12 of Falcon specification
// https://falcon-sign.info/falcon.pdf
//
// Note it's possible that caller of this function might want to fill the bytes
// array themselves, in that case, they should set template parameter's value to
// `false`. In default case, byte array can be empty and 9 random bytes will be
// sampled using Uniform Integer Distribution, while seeding Mersenne Twister
// Engine with system randomness. I strongly suggest you to look at
// `random_fill` function, that's invoked below.
template<const bool sample = true>
static inline uint32_t
base_sampler(std::array<uint8_t, 9> bytes = {})
{
  if constexpr (sample) {
    random_fill(bytes.data(), bytes.size());
  }

  const u72::u72_t u = u72::u72_t::from_le_bytes(std::move(bytes));

  uint32_t z0 = 0u;
  for (size_t i = 0; i < 18; i++) {
    z0 = z0 + 1u * (u < RCDT[i]);
  }

  return z0;
}

// Given two 64 -bit unsigned integer operands, this routine multiplies them
// such that high and low 64 -bit limbs of 128 -bit result in accessible.
//
// Note, returned pair holds high 64 -bits of result first and then remaining
// low 64 -bits are kept.
//
// Taken from
// https://github.com/itzmeanjan/rescue-prime/blob/faa22ec/include/ff.hpp#L15-L70
static inline constexpr std::pair<uint64_t, uint64_t>
full_mul_u64(const uint64_t lhs, const uint64_t rhs)
{
#if defined __aarch64__ && __SIZEOF_INT128__ == 16
  // Benchmark results show that only on aarch64 CPU, if __int128 is supported
  // by the compiler, it outperforms `else` code block, where manually high and
  // low 64 -bit limbs are computed.

  using uint128_t = unsigned __int128;

  const auto a = static_cast<uint128_t>(lhs);
  const auto b = static_cast<uint128_t>(rhs);
  const auto c = a * b;

  return std::make_pair(static_cast<uint64_t>(c >> 64),
                        static_cast<uint64_t>(c));

#else
  // On x86_64 targets, following code block always performs better than above
  // code block - as per benchmark results.

  const uint64_t lhs_hi = lhs >> 32;
  const uint64_t lhs_lo = lhs & 0xfffffffful;

  const uint64_t rhs_hi = rhs >> 32;
  const uint64_t rhs_lo = rhs & 0xfffffffful;

  const uint64_t hi = lhs_hi * rhs_hi;   // high 64 -bits
  const uint64_t mid0 = lhs_hi * rhs_lo; // mid 64 -bits ( first component )
  const uint64_t mid1 = lhs_lo * rhs_hi; // mid 64 -bits ( second component )
  const uint64_t lo = lhs_lo * rhs_lo;   // low 64 -bits

  const uint64_t mid0_hi = mid0 >> 32;          // high 32 -bits of mid0
  const uint64_t mid0_lo = mid0 & 0xfffffffful; // low 32 -bits of mid0
  const uint64_t mid1_hi = mid1 >> 32;          // high 32 -bits of mid1
  const uint64_t mid1_lo = mid1 & 0xfffffffful; // low 32 -bits of mid1

  const uint64_t t0 = lo >> 32;
  const uint64_t t1 = t0 + mid0_lo + mid1_lo;
  const uint64_t carry = t1 >> 32;

  // res = lhs * rhs | res is a 128 -bit number
  //
  // assert res = (res_hi << 64) | res_lo
  const uint64_t res_hi = hi + mid0_hi + mid1_hi + carry;
  const uint64_t res_lo = lo + (mid0_lo << 32) + (mid1_lo << 32);

  return std::make_pair(res_hi, res_lo);

#endif
}

// Given an unsigned 126 -bit result held using two 64 -bit unsigned integers (
// first high 62 -bits and low 64 -bits ), this routine extracts out and returns
// top 63 -bits of result.
static inline uint64_t
top_63_bits(std::pair<uint64_t, uint64_t> v)
{
  constexpr uint64_t mask = (1ul << 62) - 1ul;
  return ((v.first & mask) << 1) | (v.second >> 63);
}

// Routine for computing integral approximations of
//
// 2^63 * ccs * e^−x | x ∈ [0, ln(2)] , ccs ∈ [0, 1]
//
// This is an implementation of algorithm 13, described on page 42 of Falcon
// specification https://falcon-sign.info/falcon.pdf
static inline uint64_t
approx_exp(const double x, const double ccs)
{
  uint64_t y = C[0];
  uint64_t z = static_cast<uint64_t>(std::floor(9223372036854775808. * x));

  for (size_t u = 1; u < 13; u++) {
    const auto t0 = full_mul_u64(z, y);
    const auto t1 = top_63_bits(t0);
    y = C[u] - t1;
  }

  z = static_cast<uint64_t>(std::floor(9223372036854775808. * ccs));
  const auto t0 = full_mul_u64(z, y);
  y = top_63_bits(t0);

  return y;
}

}
