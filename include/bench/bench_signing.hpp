#pragma once
#include "falcon.hpp"
#include "prng.hpp"
#include <benchmark/benchmark.h>
#include <cassert>

// Benchmark Falcon PQC suite implementation
namespace bench_falcon {

// Benchmark Falcon{512, 1024} message signing algorithm, emulating only single
// message is signed with secret key.
//
// Note, this signing API builds matrix B and falcon tree T, everytime it's
// asked to sign a message, by decoding Falcon secret key.
template<const size_t N>
void
sign_single(benchmark::State& state)
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
  prng::prng_t rng;

  falcon::keygen<N>(pkey, skey);
  rng.read(msg, mlen);

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

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));

  const bool verified = falcon::verify<N>(pkey, msg, mlen, sig);

  std::free(pkey);
  std::free(skey);
  std::free(sig);
  std::free(msg);

  assert(verified);
}

// Benchmark Falcon{512, 1024} message signing algorithm, emulating many
// messages are consecutively signed with same secret key.
//
// Note, this signing API doesn't build matrix B and falcon tree T, everytime
// it's asked to sign a message, rather it keeps them loaded in memory. So this
// benchmark result should be faster compared to above `sign_single` benchmark
// result.
template<const size_t N>
void
sign_many(benchmark::State& state)
  requires((N == 512) || (N == 1024))
{
  const size_t mlen = state.range();

  constexpr size_t siglen = falcon_utils::compute_sig_len<N>();
  constexpr size_t matblen = 2 * 2 * N; // 2x2 matrix B = [[g, -f], [G, -F]]
  constexpr size_t ftlen = (log2<N>() + 1) * (1ul << log2<N>()); // 2^k * (k+1)

  // see table 3.3 of falcon specification
  constexpr double σ_values[]{ 165.736617183, 168.388571447 };
  constexpr double σ = σ_values[N == 1024];

  // see table 3.3 of falcon specification
  constexpr int32_t β2_values[]{ 34034726, 70265242 };
  constexpr int32_t β2 = β2_values[N == 1024];

  auto B = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * matblen));
  auto T = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * ftlen));
  auto h = static_cast<ff::ff_t*>(std::malloc(sizeof(ff::ff_t) * N));
  auto sig = static_cast<uint8_t*>(std::malloc(siglen));
  auto msg = static_cast<uint8_t*>(std::malloc(mlen));
  prng::prng_t rng;

  keygen::keygen<N>(B, T, h, σ, rng);
  rng.read(msg, mlen);

  for (auto _ : state) {
    falcon::sign<N>(B, T, msg, mlen, sig, rng);

    benchmark::DoNotOptimize(B);
    benchmark::DoNotOptimize(T);
    benchmark::DoNotOptimize(msg);
    benchmark::DoNotOptimize(mlen);
    benchmark::DoNotOptimize(sig);
    benchmark::DoNotOptimize(rng);
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));

  const bool verified = verification::verify<N, β2>(h, msg, mlen, sig);

  std::free(B);
  std::free(T);
  std::free(h);
  std::free(sig);
  std::free(msg);

  assert(verified);
}

}
