#pragma once
#include "common.hpp"
#include "u72.hpp"

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

}
