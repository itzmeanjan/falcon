# falcon
Fast Fourier Lattice-based Compact Signatures over NTRU - NIST PQC Digital Signature Algorithm

## Overview

Falcon is one of those post-quantum digital signature algorithms ( DSA ), which are selected by NIST for standardization purpose. Falcon is a hash-and-sign lattice-based signature scheme, built on top of NTRU lattices. Falcon signature scheme can be described with following

**FALCON = GPV Framework + NTRU Lattices + Fast Fourier Sampling**

Falcon DSA offers following algorithms

- Keypair generation
- Signing of message
- Verification of signature

Algorithm | Input | Output | What does it do ?
--- | --- | --- | --:
`keygen` | | Public key and Private key | Generates a new keypair by solving NTRU equation.
`sign` | Private key, M -bytes message | Compressed Signature | Given private key and a message, this routine signs message using Falcon Tree and ffSampling, while finally generating compressed signature bytes.
`verify` | Public key, M -bytes message and compressed signature | Boolean flag denoting success ( true ) or failure ( false ) case of signature verification procedure | Given public key, a message ( which was signed ) and respective compressed signature, this routine verifies signature, returning truth value in case of successful signature verification.

Here I'm maintaining a header-only C++ library implementing Falcon512 and Falcon1024 post-quantum digital signature algorithms, which is fairly easy to use, see [below](#usage).

Falcon specification, which I thoroughly followed during this work, can be found @ https://falcon-sign.info/falcon.pdf. I suggest you go through that for having an in-depth understanding of Falcon DSA.

> **Note** You may wish to follow progress of NIST PQC standardization effort @ https://csrc.nist.gov/projects/post-quantum-cryptography.

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

- SHAKE256 XOF from `sha3` library is used for hashing message ( input to sign/ verify routine ) to a lattice point. `sha3` itself is a zero-dependency, header-only C++ library which is pinned to some specific commit using git submodule. For importing `sha3`, issue following commands after cloning this repository.

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

- For benchmarking Falcon key generation/ signing/ verification routines, targeting CPU systems, you'll need `google-benchmark` header files and library (globally) installed. Follow https://github.com/google/benchmark/tree/b111d01c#installation for installation guideline.

## Testing

For ensuring functional correctness of Falcon implementation ( along with its components such as `NTRUGen`, `NTRUSolve`, `samplerZ` or `ffSampling` etc. ) issue

> **Warning** This implementation of Falcon is not yet tested to be **conformant** with NIST submission of Falcon - that's because I've not yet tested it with **K**nown **A**nswer **T**ests which are present in Falcon submission package.

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

## Usage

`falcon` is a minimal, header-only C++ library which is pretty easy to use.

1. Clone Falcon repository
2. Fetch `sha3` git submodule dependency, see [here](#prerequisites)
3. Install GNU MP, see [here](#prerequisites)
4. Write program which makes use of Falcon key generation/ signing/ verification API, while including `include/falcon.hpp` and using functions living inside `falcon::` namespace.
5. Finally when compiling program, let your compiler know where it can find Falcon, Sha3 and GMP headers along with libraries while passing proper flags to linker, see [this](./Makefile) build recipe.

Following namespaces are of your interest.

Namespace | Header | What can it do for you ?
--- | --- | --:
`falcon::` | `include/falcon.hpp` | Includes key generation, signing and verification algorithm definitions. **Just including this header should give you access to almost all namespaces**
`falcon_utils::` | `include/utils.hpp` | Can help you in compile-time computing length of Falcon{512, 1024} public/ private key and signature.
`decoding::` | `include/decoding.hpp` | Holds definitions for decoding public key, private key and compressed signature.

---

**Using Falcon DSA API**

- First step in using Falcon DSA is generating a new keypair. 

> **Note** Generated keypair is encoded as byte array, so one may just write it to a file.

```cpp
// Falcon512 key generation

#include "falcon.hpp"

constexpr size_t N = 512;

uint8_t pkey[falcon_utils::compute_pkey_len<N>()]{};
uint8_t skey[falcon_utils::compute_skey_len<N>()]{};

falcon::keygen<N>(pkey, skey);
```

- Once keypairs are generated they can be used for signing messages. There are broadly two scenarios related to signing
    - Private key is loaded from disk to sign a single message.
    - Private key is loaded from disk and kept in handy format so that many consecutive messages can be signed. 

- Let's take the first scenario, where only one message needs to signed. In this case private key will be first loaded from disk and decoded into three polynomials f, g and F. Then NTRU equation will be used for recomputing value of polynomial G. These four polynomils will be used for computing a 2x2 matrix B ( in its FFT representation ) s.t. $B_{2*2} = [[g, -f], [G, -F]]$ and a Falcon Tree T ( also in its FFT representation ). Now we're ready to sign the message.

```cpp
// Falcon512 sign single message

#include "falcon.hpp"

constexpr size_t N = 512;
prng::prng_t rng;

uint8_t msg[32];
uint8_t sig[falcon_utils::compute_sig_len<N>()]{};

rng.read(msg, sizeof(msg));
const bool _signed = falcon::sign<N>(skey, msg, sizeof(msg), sig);

assert(_signed);
```

- If interested in signing many messages, one after another, then it's a better idea to precompute 2x2 matrix B and Falcon Tree T from private key.

```cpp
// Falcon512 sign many messages

#include "falcon.hpp"

constexpr size_t N = 512;
prng::prng_t rng;

int32_t f[N];            
int32_t g[N];
int32_t F[N];
int32_t G[N];
fft::cmplx B[2 * 2 * N];
fft::cmplx T[(1ul << log2<N>()) * (log2<N>() + 1)];

const bool _decoded = decoding::decode_skey<N>(skey, f, g, F);
assert(_decoded);

falcon::recompute_G<N>(f, g, F, G);
falcon::compute_matrix_B<N>(f, g, F, G, B);
falcon::compute_falcon_tree<N>(B, T);

uint8_t msg[32];
uint8_t sig[falcon_utils::compute_sig_len<N>()]{};

rng.read(msg, sizeof(msg));
falcon::sign<N>(B, T, msg, msglen, sig, rng);
```

- The remaining part of using Falcon DSA is verifying signature using public key. 

```cpp
// Falcon512 signature verification

#include "falcon.hpp"

constexpr size_t N = 512;

const bool _verified = falcon::verify<N>(pkey, msg, msglen, sig);
assert(_verified);
```

--- 

I strongly advise you to go through following examples demonstrating usage of Falcon key generation/ signing/ verification API.

- [Sign a single message](./example/sign_one.cpp)
- [Sign many messages](./example/sign_many.cpp)

Here's an example showing how to compile and run these examples.

```bash
$ clang++ -std=c++20 -Wall -O3 -march=native -mtune=native -I include/ -I sha3/include/ example/sign_one.cpp -lgmpxx -lgmp && ./a.out

Falcon512 (Sign Single Message)

Public Key : 09dae803a400a1707302675a49e4749f2ec66b1f3990a2647498158e3713991df8b32553999b61d79fbc1bc2934788a8d532fec9ca059018853568ea36e55a60cd8cdcba48cc8d5b0480a08c76619c89b8f7a1f046b38af552a0350b40c9c6423d407f911afd7241f5bf9b44bbd531955a5e50e9f512121157c8b3b192b6bfbf89da57885c58d9a5a1f1b93042de8a48b397f05a36106a057b952fb886a88102b0770f5e02242e694625e63a492d247433acd88810082399c9cdb792bd4dde5864857b5ab8ff93943a62294efa86b7306244860907fc4493f859c2df05c64cd68c5ba5f409886255b8eb3371f141978f9e8e3297891f961f8883c16870d2415682435a2c9f4f6b667b7d62725118f8016238fbec7db4f7513c141d4ab3d9987889538c32a2d8bbb45b144334cd04b2af374bf6e819c58846ab1ed5a854dfd098ee60a16aac4e38e53eaa514c159536b0b9b5460ac38fc41e88d58985014a729a11a7f0fefcb8542a64e03d6230fd5ca38a2e829f330e8f06b95186cae51d88635992b3c055f3f9fd0f016462e616954e5180c8db0ed099b9aaa59b77fc9f76881e780e845506e20397fc4c189ba9f8cb591ea5f2a99771eb264ac988ec19161dbd999b0b0ba23138ca537d2ab156b49d6eaca2074e427f633380f9da6ee5eaeec6c39c6ca927ce07d425b592ee774776ea70168323e442023b15ad53d069d8ad4352160b664d0722a7c464fccd2403ecb3b88698464c6985131d56a175c2ec62caa8a7aaa28237a2f557336fbb16c51453a61c5aa0c6925157e30f2528026eca210c03d9d138cfc5643604cd4d9691f966ba4435ce12d672432894dea1aad96fc5a9ab185bc087a43df51679152159052ad2a11866d66e1ad3bbe1f4f600c8a7c044c41a95b4c43093b3a95442c2e02a5ff85de52f28b41fcd95a5902dc25999102c1dd94a99fe006d9edd5a749c8b49249cd0f25a4babe994f983582ddea335b581d439080f08126b141ef55ce48a147187529c52c8dda15a90d494984e21460b418d3369e569dba08c5415f2903be2e85a944e662e046fb52b961c0e1f33b10ba5361ea5c8f9bb1ebe5e096a63fc80a35b226d612a21b91d8f0929d481361c8ad72b34462791e744c1e3922163545d5912848b230548781c4ebd574d178a0e64984949f5d49fb15863e6d9d9b4195270acc6b21035268810498a5754a9c7010057870e08c25035401f367b58a65bab715d765f96c041144419128f29b634f267
Secret Key : 593e7104fd10e405e10ffc3ef07cadeb3af107ff7e00840ff400e10703c223c35efc421ff883bf03fdbf0ffe400840df03c01d003dd207fe10fc4771f4fcff07ffc0eb454d003e6f14bd20003caf2b83f0ebfb0f108befffc100087c20fcc300fc41f0130511f8853f187cc10ffe3e20ff20d8f6ceffc07ffc8060f83ec20bc5100c7e620441cfcfbcbf0f3e2004011014c4ef0f81c013c5f0effceedfc5510041120c472f00c01f0c48c01b0111147c0108c14f0cc01f0401ef0f81c01f3f41fcc4e00b40e10781d0f700ce13fd2ff80220f4c04004beffff7db1fff6100807f10bbf100cc131dcc27007c341f002c1fbc6301001d01385e1effee01bcd6017848ff4819f0f7de0ffbf4f18860f003b710706f00b46e0f7c0f01b04ae23faef13810ee841f1ff44b02fc7dd0305bfff462fec0350143d80fc7f300400e0fbbfef0fc6f1ff02e00bbf10143e0ef8854edcc11e0441e0f3fe9fff41bf1f7ffff380bfef05011001f2f702410c7c11f087c01ff72f100b10f000d007becf07bd210841f1fbc7dfd741f0fb7c0edc7b20083ebffc81000404a1fbbd2f044480fcff8e0f7871187b4018ff1fe83fdf1bffeeef3dd00f417108bb21f8fdfefbb8df07c430203c010082f007fb30f40351f40200f4802e14c0100cc04f08820ff884ceefbcff03fcc0ff42e0ff0341f0bdc0f7fd00f40370e77fd1fb3ebeff7e8f0c440110410f1040700841d0137e211c3b5f0c03e20ffa7f0f8431043db1fbba0104bc3fd441effffc80fb4190efc4aeff83f0134700f846400cc8800b390fecc65ff8c1e0033611fc7b60fb433f04000010bdf0f702e00b83c0f3bb81f3c1b10b42be137f001c04e0033df1f7c2e103fdc113fc1ff8c5df1704f00bbdfff7804ffc44c00b8152f47b200084efe7bd10003b8f143f2ff03e80f841dfff3ecf1ffffffb8401f883f0077d1f20be5f1801d1ff3430143d010c02b0073be1e73f3104f94ef40690effc0e0842e10f01a107c3d11bbcf10f3de0fb44bfffbf41ec3ed00340ff0fbdf0073bffffc2af23c13e0cfdd003babf13c90fe887b00f0640d078b00f1c07d6e0e6e11ed7df02daf1c31727ee282121b4eb0df4e6071acfd3ffe720d8f3ffe8e719e5fd0702163ee20c01b1e20c090ff31c01fbfe1b2efbf0e730ccde15f2f1dd2d17f6e1e3fbe019f42ef000defcd4f7faefeee110eb17f5f801e8fb171fbddbfa06251f14d1f0f5cbfe1ffae4ec06c40217e5e4f130f9f02824d8fc10dcfe27e6f81fffff03dbfcdcf213fbdcf23ef10368f5d62e0804e410fe08c8160ad8ebf9c225fd091916efd702fef71709ecd1313bd736e501d5de302018de12f2dd0b0c22e7090212cff23509eafdfff0dee8f1f50bef1100a314e9faed2f1de009f60522d9eb14d00e14fe163a03ee310424ffe6fc131404090cdeefccfd121b2d0224f51afcc2edd301f71d06fb07dbdedf101117e714fee60bf7edeb1eda0f022d041633c63bfa03cab21800edd727d2f002c3f133dd24d222eceffb19ec3c25d2d400f4fef219fad20108e50e19f704f814fafff70cf0da02deeb04e611f51ef10cdf05e22029062de8e8fce6d82bfce5f616f21cf7e505effdf223eb0ef2f7fdf304eb25e3fb1adf04fcf9e32ceef60b071d15121afa0520f3030611eb060914f30f06e30d2fe4151b001cfcfbf4fb1b0b12db0fe5e9140021ed0d08dd15e50efaedf90311f5021a021eedf60bf125f3f7d00d07ec0f0001fd14ea1ffff4d806140d1b0715e6170bcfebe237fa21010904d919f1f1ffd8dcf6f1f131
Message    : a44b43d0d942cb65bf9f0800b7ac1de17ca6d7ba7cb96044f9320725238e10f8
Signature  : 3937578386fe6b7b65562eeb52d8e0e673ff2dd8579e77b1ef27d4533a3978584b1e5f7a4436bab0b1a6aec84e7cfc96c1d9b79be12acdb8f5bc0182cbe962344c14e296c86ae2e7da953d37640908de55721bab0d0cef78b84c262a03678aa8d74412dcf375b3a99784e4f254e31ca6ccb24c1be6d567e57d84b24bb5a6dd5537a57e56e19726ea28a44d7bee8b4a9491a5ca4dc5706ad73ffdc6ab50da6c36ac831995590d32cb72fad293bc89faa62baf0e11bc93a5a446bf8279341268745d59823f9161dbd1b30c032a90b55516455192c75eb43d0d4b0824e25c3155e6d63b47bce1a388fc8c64cd9a498c6b29345537a92a11480263a67874e264a86e46553f32fee9465d86c922b1eae223256c5edf451e399322d355f90e87d4d5080bc24dbe1bfed11f612f85e7aab330efbb8e586ff0d6d54dbc98d659a280efec55b7aa22a81297a0cda26fb0be96469687a98925a89741d469e05b0842450cfc452ef6adb90642a4ee966a5714e863b04b7dc39ba5647c8cb6c50b7e608bed4a179886b1c48151747751c3196e2d1baea25bd553b2493b446190134abc6fa37a0f55135e310bb504ebd4922e7b938553312c5e5518ee4fd99ad2dfb337246de6a662bfb7090e30cb77113b56adf2c7763b83874924a8ad6fedc556b954ad78de99997ebb54de54526e670bcf3e240bc18395953fff41352904cbaf8668785db8c29af8b8f9383cd9a21207ae269a9bf684e09be1dbe83d1efdb6760665eef8df7c15d71d7adb1d0a6b74d61e82fcc3a79f84e741112a51cfa42de3601254e5d7b5a9d1bf73a9da1a3e7ff2d6e5c7359e9d8e7ebca3ea8e253997f8a72bb4cd9b621eaa3493df7083c5480deebe6bd8a14db0bd7f4217b0bb5a5cbd3e93ab847c154ddc1d69f569e81a84eb6d06c3b7d000000000000000000000
Verified    : true
```
