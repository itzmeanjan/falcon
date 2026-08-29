#include "common.hpp"
#include "falcon.hpp"
#include "ntt.hpp"
#include "prng.hpp"
#include <gtest/gtest.h>

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

  constexpr size_t min_msg_len = 1;
  constexpr size_t max_msg_len = 1024;

  auto pkey = static_cast<uint8_t*>(std::malloc(pklen));
  auto skey = static_cast<uint8_t*>(std::malloc(sklen));
  auto sig = static_cast<uint8_t*>(std::malloc(siglen));
  prng::prng_t rng;

  falcon::keygen<N>(pkey, skey);

  for (size_t mlen = min_msg_len; mlen <= max_msg_len; mlen++) {
    auto msg = static_cast<uint8_t*>(std::malloc(mlen));
    rng.read(msg, mlen);

    const bool _signed = falcon::sign<N>(skey, msg, mlen, sig);
    const bool _verified = falcon::verify<N>(pkey, msg, mlen, sig);

    EXPECT_TRUE(_signed && _verified);
    std::free(msg);
  }

  std::free(pkey);
  std::free(skey);
  std::free(sig);
}

TEST(Falcon, KeygenSignVerify)
{
  test_keygen_sign_verify<ntt::FALCON512_N>();
  test_keygen_sign_verify<ntt::FALCON1024_N>();
}

// Generates a random Falcon{512, 1024} keypair, signs one message and attempts to verify that
// signature against a DIFFERENT message - which must always fail.
//
// A round-trip test cannot reach this path. Verification computes s1 = c - s2*h mod q and checks
// that the squared norm of (s1, s2) is within β2; for a genuine signature that norm is small, so
// the accumulator is never stressed. It is only a signature that must be REJECTED whose norm runs
// into the billions, and that is where a too-narrow accumulator wraps negative and passes the
// bound. Hence a negative test, and hence many trials: the wrap depends on the coefficients, so a
// single pair proves little.
template<const size_t N>
void
test_wrong_message_rejected()
  requires((N == 512) || (N == 1024))
{
  constexpr size_t pklen = falcon_utils::compute_pkey_len<N>();
  constexpr size_t sklen = falcon_utils::compute_skey_len<N>();
  constexpr size_t siglen = falcon_utils::compute_sig_len<N>();

  constexpr size_t mlen = 32;
  // 256, not a handful. The acceptance rate depends on the compiler: the defect is signed-integer
  // overflow, so an optimiser is entitled to assume it cannot happen, and under clang at -O2 and
  // above the falcon-512 symptom disappears entirely while falcon-1024 drops to roughly 3 in 40.
  // At that rate a 32-trial test passes on broken code about 1 run in 12. This many trials makes
  // detection near-certain on every compiler and optimisation level measured.
  constexpr size_t trials = 256;

  auto pkey = static_cast<uint8_t*>(std::malloc(pklen));
  auto skey = static_cast<uint8_t*>(std::malloc(sklen));
  auto sig = static_cast<uint8_t*>(std::malloc(siglen));
  auto msg = static_cast<uint8_t*>(std::malloc(mlen));
  auto other = static_cast<uint8_t*>(std::malloc(mlen));
  prng::prng_t rng;

  falcon::keygen<N>(pkey, skey);

  for (size_t i = 0; i < trials; i++) {
    rng.read(msg, mlen);
    rng.read(other, mlen);

    const bool _signed = falcon::sign<N>(skey, msg, mlen, sig);
    ASSERT_TRUE(_signed);

    // control: the signature must verify against the message it was made over, otherwise a
    // rejection below would prove nothing about the wrong-message case.
    EXPECT_TRUE(falcon::verify<N>(pkey, msg, mlen, sig));

    EXPECT_FALSE(falcon::verify<N>(pkey, other, mlen, sig));
  }

  std::free(pkey);
  std::free(skey);
  std::free(sig);
  std::free(msg);
  std::free(other);
}

TEST(Falcon, WrongMessageRejected)
{
  test_wrong_message_rejected<ntt::FALCON512_N>();
  test_wrong_message_rejected<ntt::FALCON1024_N>();
}
