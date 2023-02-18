#include "falcon.hpp"
#include <cassert>
#include <iostream>

// Compile it with
//
// clang++ -std=c++20 -Wall  -O3 -march=native -mtune=native -I include/ -I
// sha3/include/ example/sign_one.cpp -lgmpxx -lgmp
int
main()
{
  // Try changing N to 1024 if interested in using FALCON1024
  constexpr size_t N = 512;

  constexpr size_t pklen = falcon_utils::compute_pkey_len<N>();
  constexpr size_t sklen = falcon_utils::compute_skey_len<N>();
  constexpr size_t siglen = falcon_utils::compute_sig_len<N>();
  constexpr size_t msglen = 32;

  auto pkey = static_cast<uint8_t*>(std::malloc(pklen));
  auto skey = static_cast<uint8_t*>(std::malloc(sklen));
  auto sig = static_cast<uint8_t*>(std::malloc(siglen));
  auto msg = static_cast<uint8_t*>(std::malloc(msglen));

  // random message to be signed
  prng::prng_t rng;
  rng.read(msg, msglen);

  // generate FALCON512 keypair
  falcon::keygen<N>(pkey, skey);
  // sign message using FALCON512 private key
  const bool _signed = falcon::sign<N>(skey, msg, msglen, sig);
  // verify message using FALCON512 public key
  const bool _verified = falcon::verify<N>(pkey, msg, msglen, sig);

  std::cout << "Falcon" << N << "\n\n";
  std::cout << "Public Key : " << to_hex(pkey, pklen) << "\n";
  std::cout << "Secret Key : " << to_hex(skey, sklen) << "\n";
  std::cout << "Message    : " << to_hex(msg, msglen) << "\n";
  std::cout << "Signature  : " << to_hex(sig, siglen) << "\n";
  std::cout << "Verified    : " << std::boolalpha << _verified << "\n";

  std::free(pkey);
  std::free(skey);
  std::free(sig);
  std::free(msg);

  assert(_signed && _verified);

  return 0;
}
