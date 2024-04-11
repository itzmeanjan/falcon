#pragma once
#include "common.hpp"
#include "falcon.hpp"
#include "prng.hpp"
#include <cassert>

// Test functional correctness of Falcon PQC suite implementation
namespace test_falcon {

// Generates random Falcon{512, 1024} keypair, takes random message bytes of
// length ∈ [0, 1024), signs message and attempts to verify - all should work.
template<const size_t N>
void
test_keygen_sign_verify()
  requires((N == 512) || (N == 1024))
{
  constexpr size_t pklen = falcon_utils::compute_pkey_len<N>();
  constexpr size_t sklen = falcon_utils::compute_skey_len<N>();
  constexpr size_t siglen = falcon_utils::compute_sig_len<N>();

  auto pkey = static_cast<uint8_t*>(std::malloc(pklen));
  auto skey = static_cast<uint8_t*>(std::malloc(sklen));
  auto sig = static_cast<uint8_t*>(std::malloc(siglen));
  prng::prng_t rng;

  falcon::keygen<N>(pkey, skey);

  for (size_t mlen = 0; mlen < 1024; mlen++) {
    auto msg = static_cast<uint8_t*>(std::malloc(mlen));
    rng.read(msg, mlen);

    const bool _signed = falcon::sign<N>(skey, msg, mlen, sig);
    const bool _verified = falcon::verify<N>(pkey, msg, mlen, sig);

    assert(_signed && _verified);
    std::free(msg);
  }

  std::free(pkey);
  std::free(skey);
  std::free(sig);
}

}
