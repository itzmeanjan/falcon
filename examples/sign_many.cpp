#include "falcon.hpp"
#include <cassert>
#include <iostream>

// Compile it with
//
// clang++ -std=c++20 -Wall  -O3 -march=native -mtune=native -I include/ -I
// sha3/include/ example/sign_many.cpp -lgmpxx -lgmp
int
main()
{
  // Try changing N to 1024 if interested in using FALCON1024
  constexpr size_t N = 512;

  constexpr size_t pklen = falcon_utils::compute_pkey_len<N>();
  constexpr size_t sklen = falcon_utils::compute_skey_len<N>();
  constexpr size_t siglen = falcon_utils::compute_sig_len<N>();

  // Sign msgcnt -many messages s.t. each message is msglen -bytes wide
  constexpr size_t msglen = 32;
  constexpr size_t msgcnt = 4;

  // Complex numbers required for representing Falcon Tree of height log2(N)
  constexpr size_t treelen = (1ul << log2<N>()) * (log2<N>() + 1);

  auto pkey = static_cast<uint8_t*>(std::malloc(pklen));
  auto skey = static_cast<uint8_t*>(std::malloc(sklen));
  auto sig = static_cast<uint8_t*>(std::malloc(siglen));
  auto msg = static_cast<uint8_t*>(std::malloc(msglen));

  // generate FALCON512 keypair
  falcon::keygen<N>(pkey, skey);

  std::cout << "Falcon" << N << " (Sign Many Messages)\n\n";
  std::cout << "Public Key : " << to_hex(pkey, pklen) << "\n";
  std::cout << "Secret Key : " << to_hex(skey, sklen) << "\n\n\n";

  // ----- Prepare private key in form of 2x2 matrix B and Falcon Tree T -----

  prng::prng_t rng;        // Source of randomness
  int32_t f[N];            // Part of Falcon512 secret key
  int32_t g[N];            // Part of Falcon512 secret key
  int32_t F[N];            // Part of Falcon512 secret key
  int32_t G[N];            // Part of Falcon512 secret key (computed from f,g,F)
  fft::cmplx B[2 * 2 * N]; // 2x2 matrix B = [[g, -f], [G, -F]]
  fft::cmplx T[treelen];   // Falcon Tree

  // Try to decode secret key and obtain f, g and F
  const bool _decoded = decoding::decode_skey<N>(skey, f, g, F);
  // If fails to decode, abort !
  assert(_decoded);

  // Compute G from f, g, F ( by solving NTRU equation )
  falcon::recompute_G<N>(f, g, F, G);
  // Compute 2x2 matrix B = [[g, -f], [G, -F]] ( in FFT form )
  falcon::compute_matrix_B<N>(f, g, F, G, B);
  // Compute Falcon Tree in its FFT form
  falcon::compute_falcon_tree<N>(B, T);

  // ----- Now private key is represented in form of B and T -----

  for (size_t i = 0; i < msgcnt; i++) {
    // now generate a random message to be signed
    rng.read(msg, msglen);

    // Use precomputed 2x2 matrix B ( in FFT form ), falcon tree ( in FFT form )
    // to sign message
    falcon::sign<N>(B, T, msg, msglen, sig, rng);
    // Verify message signature by just using public key
    const bool _verified = falcon::verify<N>(pkey, msg, msglen, sig);
    assert(_verified);

    std::cout << "Message    : " << to_hex(msg, msglen) << "\n";
    std::cout << "Signature  : " << to_hex(sig, siglen) << "\n";
    std::cout << "Verified    : " << std::boolalpha << _verified << "\n\n\n";
  }

  std::free(pkey);
  std::free(skey);
  std::free(sig);
  std::free(msg);

  return 0;
}
