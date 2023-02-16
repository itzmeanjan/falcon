#pragma once
#include "common.hpp"
#include "prng.hpp"
#include "u72.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <utility>

// Sampler over the Integers
namespace samplerz {

// = math.log(2)
constexpr double LN2 = 0.6931471805599453;

// = 1/ math.log(2)
constexpr double INV_LN2 = 1. / LN2;

// See table 3.3 of Falcon specification https://falcon-sign.info/falcon.pdf
constexpr double FALCON512_σ_min = 1.277833697;

// See table 3.3 of Falcon specification https://falcon-sign.info/falcon.pdf
constexpr double FALCON1024_σ_min = 1.298280334;

// See table 3.3 of Falcon specification https://falcon-sign.info/falcon.pdf
constexpr double σ_max = 1.8205;

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
static inline uint32_t
base_sampler(std::array<uint8_t, 9>&& bytes)
{
  const u72::u72_t u = u72::u72_t::from_le_bytes(std::move(bytes));

  uint32_t z0 = 0u;
  for (size_t i = 0; i < 18; i++) {
    z0 = z0 + 1u * (u < RCDT[i]);
  }

  return z0;
}

// BaseSampler routine as defined in algorithm 12 of Falcon specification
// https://falcon-sign.info/falcon.pdf s.t. 72 uniform random bits are sampled
// from SHAKE256 based PRNG ( which is a parameter of this function ).
static inline uint32_t
base_sampler(prng::prng_t& rng)
{
  std::array<uint8_t, 9> bytes;
  rng.read(bytes.data(), bytes.size());

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

// Computes a single bit ( = 1 ) with probability ≈ ccs * e^−x | ccs, x >= 0
//
// This is an implementation of algorithm 14, described on page 43 of Falcon
// specification https://falcon-sign.info/falcon.pdf s.t. 8 uniform random bits
// are sampled using SHAKE256 based PRNG.
static inline uint8_t
ber_exp(const double x, const double ccs, prng::prng_t& rng)
{
  const double s = std::floor(x * INV_LN2);
  const double r = x - s * LN2;
  const uint64_t s_ = std::min<uint64_t>(static_cast<uint64_t>(s), 63ul);
  const uint64_t z = (2 * approx_exp(r, ccs) - 1) >> s_;

  int32_t w = 0;
  int64_t i = 64l;
  do {
    i = i - 8l;

    uint8_t t0;
    rng.read(&t0, sizeof(t0));

    w = static_cast<int32_t>(t0) - static_cast<int32_t>((z >> i) & 0xfful);
  } while ((w == 0) && (i > 0l));

  return w < 0;
}

// Computes a single bit ( = 1 ) with probability ≈ ccs * e^−x | ccs, x >= 0
//
// This is an implementation of algorithm 14, described on page 43 of Falcon
// specification https://falcon-sign.info/falcon.pdf
//
// Note, there's another function with almost similar signature, but that one
// doesn't take any random bytes array, rather samples randomness itself.
// Whereas this routine expects you to also pass pointer to some memory location
// where consecutive memory addresses hold `rblen` -many random sampled bytes.
// This routine takes randomness from that array and also return how many random
// bytes it had to use to finish executing the body of the do-while loop, so
// that next user of random bytes can just skip forward those many bytes.
static inline std::pair<uint8_t, size_t>
ber_exp(const double x,
        const double ccs,
        const uint8_t* const rbytes,
        const size_t rblen)
{
  const double s = std::floor(x * INV_LN2);
  const double r = x - s * LN2;
  const uint64_t s_ = std::min<uint64_t>(static_cast<uint64_t>(s), 63ul);
  const uint64_t z = (2 * approx_exp(r, ccs) - 1) >> s_;

  size_t ridx = 0;
  int32_t w = 0;
  int64_t i = 64l;

  do {
    i = i - 8l;

    const uint8_t t0 = rbytes[ridx++];
    w = static_cast<int32_t>(t0) - static_cast<int32_t>((z >> i) & 0xfful);
  } while ((w == 0) && (i > 0l) && (ridx < rblen));

  return std::make_pair(w < 0, ridx);
}

// Given floating point arguments μ, σ' | σ' ∈ [σ_min, σ_max], integer z ∈ Z,
// sampled from a distribution very close to D_{Z, μ, σ′}, following algorithm
// 15 of Falcon specification https://falcon-sign.info/falcon.pdf s.t. all
// random bits are sampled from a SHAKE256 based PRNG.
static inline int32_t
samplerz(const double μ,
         const double σ_prime,
         const double σ_min,
         prng::prng_t& rng)
{
  const double r = μ - std::floor(μ);
  const double ccs = σ_min / σ_prime;

  const double t0 = 1. / (2. * σ_prime * σ_prime);
  constexpr double t1 = 1. / (2. * σ_max * σ_max);

  while (true) {
    const auto z0 = static_cast<int32_t>(base_sampler(rng));

    uint8_t v;
    rng.read(&v, sizeof(v));

    const auto b = v & 0b1;
    const auto z = static_cast<double>(b + (2 * b - 1) * z0);

    const auto t2 = z - r;
    const auto t3 = t2 * t2;
    const auto t4 = t3 * t0;

    const auto t5 = static_cast<double>(z0 * z0);
    const auto t6 = t5 * t1;

    const auto x = t4 - t6;
    const auto t7 = ber_exp(x, ccs, rng);
    if (t7 == 1) {
      return static_cast<int32_t>(z + std::floor(μ));
    }
  }
}

// Given floating point arguments μ, σ' | σ' ∈ [σ_min, σ_max], integer z ∈ Z,
// sampled from a distribution very close to D_{Z, μ, σ′}, following algorithm
// 15 of Falcon specification https://falcon-sign.info/falcon.pdf
//
// Note, there's another function with almost similar signature, but that one
// doesn't take any random bytes array, rather samples randomness itself.
// Whereas this routine expects you to also pass pointer to some memory location
// where consecutive memory addresses hold `rblen` -many random sampled bytes.
// This routine takes randomness from that array and also return how many random
// bytes it had to use to finish executing the body of the while loop. This
// routine is written such that I can easily write test cases using KATs (known
// answer tests) suppiled with Falcon's NIST submission, for easing correct
// implementation of SamplerZ routine.
static inline std::pair<int32_t, size_t>
samplerz(const double μ,
         const double σ_prime,
         const double σ_min,
         const uint8_t* const rbytes,
         const size_t rblen)
{
  const double r = μ - std::floor(μ);
  const double ccs = σ_min / σ_prime;

  const double t0 = 1. / (2. * σ_prime * σ_prime);
  constexpr double t1 = 1. / (2. * σ_max * σ_max);

  size_t ridx = 0;
  int32_t ret_z = 0;

  while (ridx < rblen) {
    std::array<uint8_t, 9> tmp{};
    std::memcpy(tmp.data(), rbytes + ridx, 9);
    std::reverse(tmp.begin(), tmp.end());
    ridx += 9;

    const auto z0 = static_cast<int32_t>(base_sampler(std::move(tmp)));
    const auto b = rbytes[ridx++] & 0b1;
    const auto z = static_cast<double>(b + (2 * b - 1) * z0);

    const auto t2 = z - r;
    const auto t3 = t2 * t2;
    const auto t4 = t3 * t0;

    const auto t5 = static_cast<double>(z0 * z0);
    const auto t6 = t5 * t1;

    const auto x = t4 - t6;
    const auto [t7, ulen] = ber_exp(x, ccs, rbytes + ridx, rblen - ridx);
    ridx += ulen;
    if (t7 == 1) {
      ret_z = static_cast<int32_t>(z + std::floor(μ));
      break;
    }
  }

  return std::make_pair(ret_z, ridx);
}

}
