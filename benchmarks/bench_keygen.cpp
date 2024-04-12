#include "bench_helper.hpp"
#include "falcon.hpp"
#include <benchmark/benchmark.h>

// Benchmark Falcon{512, 1024} keypair generation algorithm.
//
// Note, this keygen API neither builds matrix B nor Falcon tree T, which are
// required for Falcon message signing.
template<const size_t N>
static void
falcon_keygen(benchmark::State& state)
  requires((N == 512) || (N == 1024))
{
  constexpr size_t pklen = falcon_utils::compute_pkey_len<N>();
  constexpr size_t sklen = falcon_utils::compute_skey_len<N>();

  auto pkey = static_cast<uint8_t*>(std::malloc(pklen));
  auto skey = static_cast<uint8_t*>(std::malloc(sklen));

  for (auto _ : state) {
    falcon::keygen<N>(pkey, skey);

    benchmark::DoNotOptimize(pkey);
    benchmark::DoNotOptimize(skey);
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));

  std::free(pkey);
  std::free(skey);
}

BENCHMARK(falcon_keygen<512>)
  ->ComputeStatistics("min", compute_min)
  ->ComputeStatistics("max", compute_max);

BENCHMARK(falcon_keygen<1024>)
  ->ComputeStatistics("min", compute_min)
  ->ComputeStatistics("max", compute_max);
