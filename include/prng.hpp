#pragma once
#include "shake256.hpp"
#include <random>

// Pseudo Random Number Generator
namespace prng {

// Pseudo Random Number Generator s.t. N (>0) -many random bytes are read from
// SHAKE256 XoF whose state is obtained by hashing 32 random bytes sampled from
// uniform uint8_t random number generator distribution, using Mersenne Twister
// engine, which itself is seeded with system random device ( read more @
// https://en.cppreference.com/w/cpp/numeric/random/random_device )
//
// Note, std::random_device's behaviour is implementation defined feature, so
// this PRNG implementation doesn't guarantee that it'll generate cryptographic
// secure random bytes in all possible cases.
struct prng_t
{
private:
  shake256::shake256<false> state;

public:
  inline prng_t()
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dis{};

    uint8_t seed[32];
    for (size_t i = 0; i < sizeof(seed); i++) {
      seed[i] = dis(gen);
    }

    state.hash(seed, sizeof(seed));
  }

  inline void read(uint8_t* const bytes, const size_t len)
  {
    state.read(bytes, len);
  }
};

}
