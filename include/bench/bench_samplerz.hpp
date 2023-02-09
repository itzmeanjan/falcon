#pragma once
#include "samplerz.hpp"
#include <benchmark/benchmark.h>

// Benchmark Falcon PQC suite implementation
namespace bench_falcon {

// Benchmarks Sampler over the Integers for some specific μ, σ' and σ_min
inline void
samplerz(benchmark::State& state)
{
  constexpr double μ = -91.90471153063714;
  constexpr double σ_prime = 1.7037990414754918;
  constexpr double σ_min = 1.2778336969128337;

  for (auto _ : state) {
    const auto res = samplerz::samplerz(μ, σ_prime, σ_min);

    benchmark::DoNotOptimize(res);
    benchmark::DoNotOptimize(μ);
    benchmark::DoNotOptimize(σ_prime);
    benchmark::DoNotOptimize(σ_min);
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}

}
