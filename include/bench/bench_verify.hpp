#pragma once
#include "falcon.hpp"
#include <benchmark/benchmark.h>
#include <cassert>

// Benchmark Falcon PQC suite implementation
namespace bench_falcon {

// Benchmark Falcon{512, 1024} signature verification algorithm.
//
// Note, this verification API decodes public key everytime signature
// verification is requested.
template<const size_t N>
void
verify(benchmark::State& state)
  requires((N == 512) || (N == 1024))
{
  const size_t mlen = state.range();

  constexpr size_t pklen = falcon_utils::compute_pkey_len<N>();
  constexpr size_t sklen = falcon_utils::compute_skey_len<N>();
  constexpr size_t siglen = falcon_utils::compute_sig_len<N>();

  auto pkey = static_cast<uint8_t*>(std::malloc(pklen));
  auto skey = static_cast<uint8_t*>(std::malloc(sklen));
  auto sig = static_cast<uint8_t*>(std::malloc(siglen));
  auto msg = static_cast<uint8_t*>(std::malloc(mlen));

  falcon::keygen<N>(pkey, skey);
  random_fill(msg, mlen);
  const bool _signed = falcon::sign<N>(skey, msg, mlen, sig);
  assert(_signed);

  for (auto _ : state) {
    const bool verified = falcon::verify<N>(pkey, msg, mlen, sig);

    benchmark::DoNotOptimize(verified);
    assert(verified);

    benchmark::DoNotOptimize(pkey);
    benchmark::DoNotOptimize(msg);
    benchmark::DoNotOptimize(mlen);
    benchmark::DoNotOptimize(sig);
    benchmark::ClobberMemory();
  }

  std::free(pkey);
  std::free(skey);
  std::free(sig);
  std::free(msg);
}

}
