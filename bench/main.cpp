#include "bench/bench_falcon.hpp"
#include "bench/bench_samplerz.hpp"

// register for benchmarking sampler over the integers ( Z )
BENCHMARK(bench_falcon::samplerz);

// register for benchmarking Falcon512
BENCHMARK(bench_falcon::keygen<512>);
BENCHMARK(bench_falcon::sign_single<512>)->Arg(32);
BENCHMARK(bench_falcon::sign_many<512>)->Arg(32);
BENCHMARK(bench_falcon::verify<512>)->Arg(32);

// register for benchmarking Falcon1024
BENCHMARK(bench_falcon::keygen<1024>);
BENCHMARK(bench_falcon::sign_single<1024>)->Arg(32);
BENCHMARK(bench_falcon::sign_many<1024>)->Arg(32);
BENCHMARK(bench_falcon::verify<1024>)->Arg(32);

BENCHMARK_MAIN();
