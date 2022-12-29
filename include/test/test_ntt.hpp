#pragma once
#include "ntt.hpp"
#include <cassert>
#include <cstring>

// Test functional correctness of Falcon PQC suite implementation
namespace test_falcon {

// Ensure functional correctness of (i)NTT implementation, as used in
// Falcon{512, 1024}, by executing following flow
//
// f |> NTT |> iNTT |> f' s.t. f == f'
template<const size_t lgn>
void
test_ntt()
{
  const size_t n = 1ul << lgn;

  auto* polya = static_cast<ff::ff_t*>(std::malloc(n * sizeof(ff::ff_t)));
  auto* polyb = static_cast<ff::ff_t*>(std::malloc(n * sizeof(ff::ff_t)));

  for (size_t i = 0; i < n; i++) {
    polya[i] = ff::ff_t::random();
  }
  std::memcpy(polyb, polya, n * sizeof(ff::ff_t));

  ntt::ntt<lgn>(polyb);
  ntt::intt<lgn>(polyb);

  bool flg = false;
  for (size_t i = 0; i < n; i++) {
    flg |= static_cast<bool>(polya[i].v ^ polyb[i].v);
  }

  std::free(polya);
  std::free(polyb);

  assert(!flg);
}

}
