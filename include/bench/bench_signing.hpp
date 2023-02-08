#pragma once
#include "falcon.hpp"
#include <benchmark/benchmark.h>
#include <cassert>

// Benchmark Falcon PQC suite implementation
namespace bench_falcon {

// Benchmark Falcon{512, 1024} message signing algorithm.
//
// Note, this API builds matrix B and falcon tree T, everytime it's asked to
// sign a message, by decoding Falcon secret key.
template<const size_t N>
void
sign(benchmark::State& state)
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

  for (auto _ : state) {
    const bool _signed = falcon::sign<N>(skey, msg, mlen, sig);

    benchmark::DoNotOptimize(_signed);
    assert(_signed);

    benchmark::DoNotOptimize(skey);
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
