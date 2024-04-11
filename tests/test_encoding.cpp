#include "common.hpp"
#include "decoding.hpp"
#include "encoding.hpp"
#include "ffsampling.hpp"
#include "hashing.hpp"
#include "keygen.hpp"
#include "ntt.hpp"
#include "prng.hpp"
#include <cstring>
#include <gtest/gtest.h>

// Test whether random public key ( as polynomial over Fq | q = 12289 ), can be
// correctly encoded/ decoded or not.
template<const size_t N>
static void
test_encoding_pkey()
{
  constexpr size_t pklen = falcon_utils::compute_pkey_len<N>();

  auto h = static_cast<ff::ff_t*>(std::malloc(sizeof(ff::ff_t) * N));
  auto pkey = static_cast<uint8_t*>(std::malloc(pklen));
  auto h_ = static_cast<ff::ff_t*>(std::malloc(sizeof(ff::ff_t) * N));

  for (size_t i = 0; i < N; i++) {
    h[i] = ff::ff_t::random();
  }

  encoding::encode_pkey<N>(h, pkey);
  const bool flg = decoding::decode_pkey<N>(pkey, h_);

  bool success = true;
  for (size_t i = 0; i < N; i++) {
    success &= h[i] == h_[i];
  }

  std::free(h);
  std::free(pkey);
  std::free(h_);

  EXPECT_TRUE(flg);
  EXPECT_TRUE(success);
}

TEST(Falcon, EncodeDecodePublicKey)
{
  test_encoding_pkey<ntt::FALCON512_N>();
  test_encoding_pkey<ntt::FALCON1024_N>();
}

// Test whether randomly generated ( using NTRUGen ) Falcon secret key can be
// correctly encoded/ decoded or not.
template<const size_t N>
static void
test_encoding_skey()
{
  constexpr size_t sklen = falcon_utils::compute_skey_len<N>();

  auto f = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto g = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto F = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto G = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto skey = static_cast<uint8_t*>(std::malloc(sklen));
  auto f_ = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto g_ = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto F_ = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  prng::prng_t prng;

  ntru_gen::ntru_gen<N>(f, g, F, G, prng);
  encoding::encode_skey<N>(f, g, F, skey);
  const bool flg = decoding::decode_skey<N>(skey, f_, g_, F_);

  bool success = true;
  for (size_t i = 0; i < N; i++) {
    success &= f[i] == f_[i];
    success &= g[i] == g_[i];
    success &= F[i] == F_[i];
  }

  std::free(f);
  std::free(g);
  std::free(F);
  std::free(G);
  std::free(skey);
  std::free(f_);
  std::free(g_);
  std::free(F_);

  EXPECT_TRUE(flg);
  EXPECT_TRUE(success);
}

TEST(Falcon, EncodeDecodeSecretKey)
{
  test_encoding_skey<ntt::FALCON512_N>();
  test_encoding_skey<ntt::FALCON1024_N>();
}

// Generate Falcon{512, 1024} signature ( i.e. polynomial s2 ) to ensure that it
// can be compressed successfully and if it can be compressed it should also be
// decompressed back to polynomial s2, properly.
//
// This test partially implements algorithm 10 of Falcon specification, to
// ensure correctness of signature compression/ decompression routines.
template<const size_t N>
static void
test_sig_compression(
  const double σ,     // Standard deviation ( see table 3.3 of specification )
  const double σ_min, // See table 3.3 of specification
  const double β2     // Max signature square norm ( see table 3.3 )
  )
  requires((N == 512) || (N == 1024))
{
  // See table 3.3 of the specification
  constexpr size_t siglens[]{ 666, 1280 };

  constexpr size_t ft_len = (1ul << log2<N>()) * (log2<N>() + 1);
  constexpr fft::cmplx q{ ff::Q };
  constexpr size_t mlen = 32;
  constexpr size_t salt_len = 40;
  constexpr size_t siglen = siglens[N == 1024];

  auto B = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N * 4));
  auto T = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * ft_len));
  auto h = static_cast<ff::ff_t*>(std::malloc(sizeof(ff::ff_t) * N));
  auto msg = static_cast<uint8_t*>(std::malloc(mlen));
  auto salt = static_cast<uint8_t*>(std::malloc(salt_len));
  auto c = static_cast<ff::ff_t*>(std::malloc(sizeof(ff::ff_t) * N));
  auto c_fft = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto t0 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto t1 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto z0 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto z1 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto tz0 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto tz1 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto s0 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto s1 = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  auto s2 = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto dec_s2 = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  auto sig = static_cast<uint8_t*>(std::malloc(siglen));
  auto tmp = static_cast<fft::cmplx*>(std::malloc(sizeof(fft::cmplx) * N));
  prng::prng_t prng;

  keygen::keygen<N>(B, T, h, σ, prng);
  prng.read(msg, mlen);
  prng.read(salt, salt_len);
  hashing::hash_to_point<N>(salt, salt_len, msg, mlen, c);

  for (size_t i = 0; i < N; i++) {
    c_fft[i] = fft::cmplx{ static_cast<double>(c[i].v) };
  }

  fft::fft<log2<N>()>(c_fft);
  polynomial::mul<log2<N>()>(c_fft, B + 3 * N, t0);
  polynomial::mul<log2<N>()>(c_fft, B + N, t1);

  for (size_t i = 0; i < N; i++) {
    t0[i] /= q;
    t1[i] = -(t1[i] / q);
  }

  while (1) {
    // ffSampling i.e. compute z = (z0, z1), same as line 6 of algo 10
    ffsampling::ff_sampling<N, 0, log2<N>()>(t0, t1, T, σ_min, z0, z1, prng);

    // compute tz = (tz0, tz1) = (t0 - z0, t1 - z1)
    polynomial::sub<log2<N>()>(t0, z0, tz0);
    polynomial::sub<log2<N>()>(t1, z1, tz1);

    // compute s = (s0, s1) = tz * B | tz is 1x2 and B = 2x2 ( of dimension )
    polynomial::mul<log2<N>()>(tz0, B, s0);
    polynomial::mul<log2<N>()>(tz1, B + 2 * N, tmp);
    polynomial::add_to<log2<N>()>(s0, tmp);

    polynomial::mul<log2<N>()>(tz0, B + N, s1);
    polynomial::mul<log2<N>()>(tz1, B + 3 * N, tmp);
    polynomial::add_to<log2<N>()>(s1, tmp);

    // compute (∥s0, s1∥) ^ 2
    const double sq_norm0 = ntru_gen::sqrd_norm<log2<N>()>(s0);
    const double sq_norm1 = ntru_gen::sqrd_norm<log2<N>()>(s1);
    const double sq_norm = sq_norm0 + sq_norm1;

    // check ∥s∥2 > ⌊β2⌋
    if (sq_norm <= β2) {
      fft::ifft<log2<N>()>(s1);

      for (size_t i = 0; i < N; i++) {
        s2[i] = static_cast<int32_t>(std::round(s1[i].real()));
      }

      // check if signature has been compressed
      const bool compressed = encoding::compress_sig<N, siglen>(s2, sig);
      if (compressed) {
        break;
      }
    }
  }

  // if signature can be compressed, it must also be decompressed back
  const bool decompressed = decoding::decompress_sig<N, siglen>(sig, dec_s2);

  // check if actual signature polynomial and decompressed signature polynomial
  // matches or not
  bool matches = true;
  for (size_t i = 0; i < N; i++) {
    matches &= s2[i] == dec_s2[i];
  }

  std::free(B);
  std::free(T);
  std::free(h);
  std::free(msg);
  std::free(salt);
  std::free(c);
  std::free(c_fft);
  std::free(t0);
  std::free(t1);
  std::free(z0);
  std::free(z1);
  std::free(tz0);
  std::free(tz1);
  std::free(s0);
  std::free(s1);
  std::free(s2);
  std::free(dec_s2);
  std::free(sig);
  std::free(tmp);

  EXPECT_TRUE(decompressed);
  EXPECT_TRUE(matches);
}

TEST(Falcon, SignatureCompression)
{
  test_sig_compression<ntt::FALCON512_N>(165.736617183, 1.277833697, 34034726);
  test_sig_compression<ntt::FALCON1024_N>(168.388571447, 1.298280334, 70265242);
}

// Generate random signature bytes and attempt to decompress it, see how
// decompression routine behaves. In majority of the cases it should be failing
// to decompress and if it succeeds, it should also be able to compress it
// properly - check that situation too.
template<const size_t N>
static void
test_sig_decompression()
{
  // See table 3.3 of the specification
  constexpr size_t siglens[]{ 666, 1280 };
  constexpr size_t siglen = siglens[N == 1024];

  auto sig0 = static_cast<uint8_t*>(std::malloc(siglen));
  auto sig1 = static_cast<uint8_t*>(std::malloc(siglen));
  auto s2 = static_cast<int32_t*>(std::malloc(sizeof(int32_t) * N));
  prng::prng_t prng;

  // generate random signature bytes
  prng.read(sig0, siglen);

  // attempt to decompress random signature
  const bool decompressed = decoding::decompress_sig<N, siglen>(sig0, s2);
  // which will *most* probably fail
  if (decompressed) {
    // if anyhow decompressed, we should be able to compress it too
    const bool compressed = encoding::compress_sig<N, siglen>(s2, sig1);
    EXPECT_TRUE(compressed);

    // if compressed, both should produce same signature
    //
    // note, we're skipping checking of first 41 bytes of signature, because
    // that portion consists of header byte and salt bytes
    bool matches = true;
    for (size_t i = 41; i < siglen; i++) {
      matches &= sig0[i] == sig1[i];
    }

    EXPECT_TRUE(matches);
  }

  std::free(sig0);
  std::free(sig1);
  std::free(s2);
}

TEST(Falcon, SignatureDecompression)
{
  test_sig_decompression<ntt::FALCON512_N>();
  test_sig_decompression<ntt::FALCON1024_N>();
}
