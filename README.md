# falcon
Accelerated Falcon - PQC Digital Signature Algorithm

## Prerequisites

- A C++ compiler with support for C++20 standard library

```bash
$ clang++ --version
Apple clang version 14.0.0 (clang-1400.0.29.202)
Target: x86_64-apple-darwin22.2.0
Thread model: posix
InstalledDir: /Library/Developer/CommandLineTools/usr/bin

$ g++ --version
g++ (Ubuntu 11.2.0-19ubuntu1) 11.2.0
```

- System development utilities such as `make` and `cmake`

```bash
$ make --version
GNU Make 4.3

$ cmake --version
cmake version 3.22.1
```

- SHAKE256 XOF from `sha3` library is used for hashing message ( input to sign/ verify routine ) to a lattice point. `sha3` itself is a zero-dependency, header-only C++ library which is pinned to some specific commit using git submodule. For importing `sha3` issue following commands after cloning this repository,

```bash
# Assuming repository is already cloned, if not try
#
# git clone https://github.com/itzmeanjan/falcon.git

pushd falcon
git submodule update --init
popd
```

- Falcon key generation algorithm requires us to solve NTRU equation ( see equation 3.15 of Falcon specification ) which needs arbitrary precision integer ( i.e. big integer ) arithmetic support. For that purpose I use GNU MP library's C++ interface. Find more @ https://gmplib.org/manual. Install GMP header files and library using following commands.

```bash
sudo apt-get install -y libgmp-dev # On Ubuntu/ Debian
brew install gmp                   # On MacOS
```

- For benchmarking Falcon key generation/ signing/ verification routines, targeting CPU systems, you'll need `google-benchmark` header files and library (globally) installed. Follow https://github.com/google/benchmark/tree/b111d01c#installation if not available.

## Testing

For ensuring functional correctness of Falcon implementation ( along with its components such as `NTRUGen`, `NTRUSolve`, `samplerZ` or `ffSampling` etc. ) issue

> **Warning** This implementation of Falcon is not yet tested to be **conformant** with NIST submission of Falcon --- because I've not yet tested it with **K**nown **A**nswer **T**ests which are present in Falcon submission package.

```bash
make

[test] Falcon prime field arithmetic
[test] (inverse) Number Theoretic Transform over Z_q
[test] (inverse) Fast Fourier Transform over Q
[test] Splitting and merging of polynomials in FFT form
[test] Sampler over the Integers, using KATs
[test] NTRUGen
[test] Encode/ Decode Public Key
[test] Encode/ Decode Secret Key
[test] Falcon KeyGen
[test] Fast Fourier Sampling
[test] Signature Compression/ Decompression
[test] Keygen -> Sign -> Verify
```

## Benchmarking

For benchmarking Falcon{512, 1024} key generation, signing and verification for fixed length message of 32B, issue

```bash
make benchmark
```

> **Warning** You must disable CPU frequency scaling during benchmarking; see [this](https://github.com/google/benchmark/blob/b111d01c1b4cc86da08672a68cddcbcc1cedd742/docs/user_guide.md#disabling-cpu-frequency-scaling) guide.

### On Intel(R) Core(TM) i5-8279U CPU @ 2.40GHz [ Compiled with Clang ]

```bash
2023-02-18T12:07:57+04:00
Running ./bench/a.out
Run on (8 X 2400 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB
  L1 Instruction 32 KiB
  L2 Unified 256 KiB (x4)
  L3 Unified 6144 KiB
Load Average: 2.26, 2.26, 1.98
----------------------------------------------------------------------------------------------
Benchmark                                   Time             CPU   Iterations items_per_second
----------------------------------------------------------------------------------------------
bench_falcon::keygen<512>              953701 us       952681 us            1        1.04967/s
bench_falcon::sign_single<512>/32         546 us          546 us         1304       1.83235k/s
bench_falcon::sign_many<512>/32           342 us          342 us         2082       2.92446k/s
bench_falcon::verify<512>/32             27.5 us         27.5 us        26034       36.4035k/s
bench_falcon::keygen<1024>            5385395 us      5381617 us            1       0.185818/s
bench_falcon::sign_single<1024>/32       1139 us         1136 us          632        880.013/s
bench_falcon::sign_many<1024>/32          690 us          689 us         1001       1.45054k/s
bench_falcon::verify<1024>/32            55.1 us         55.0 us        12780       18.1959k/s
```
