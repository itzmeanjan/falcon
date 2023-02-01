#pragma once
#include "encoding.hpp"
#include "keygen.hpp"

// Falcon{512, 1024} Key Generation, Signing and Verification Algorithm
namespace falcon {

// [User Friendly API] Falcon{512, 1024} key generation algorithm, which takes
// no input, does following
//
// - Generates four random polynomials f, g, F, G s.t. it solves NTRU equation
// - Computes public key h = gf^-1 mod q
// - Serializes both public key and private key as byte arrays
//
// Note, this routine doesn't compute 2x2 matrix B = [[g, -f], [G, -F]] or
// Falcon Tree T, which are required during signing period. For those, see
// keygen.hpp file's keygen() function implementation - that is an
// implementation of algorithm 4 of Falcon specification, which does no byte
// serialization for secret key or public key.
template<const size_t N>
static inline void
keygen(uint8_t* const __restrict pkey, uint8_t* const __restrict skey)
  requires((N == 512) || (N == 1024))
{
  int32_t f[N];
  int32_t g[N];
  int32_t F[N];
  int32_t G[N];
  ff::ff_t h[N];

  ntru_gen::ntru_gen<N>(f, g, F, G);
  keygen::compute_public_key<N>(f, g, h);
  encoding::encode_pkey<N>(h, pkey);
  encoding::encode_skey<N>(f, g, F, skey);
}

}
