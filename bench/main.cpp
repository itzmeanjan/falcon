#include "bench/bench_falcon.hpp"

// register for benchmarking Falcon512
BENCHMARK(bench_falcon::keygen<512>);
BENCHMARK(bench_falcon::sign<512>)->Arg(32);
BENCHMARK(bench_falcon::verify<512>)->Arg(32);

// register for benchmarking Falcon1024
BENCHMARK(bench_falcon::keygen<1024>);
BENCHMARK(bench_falcon::sign<1024>)->Arg(32);
BENCHMARK(bench_falcon::verify<1024>)->Arg(32);

BENCHMARK_MAIN();
