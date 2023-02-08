#include "test/test_falcon.hpp"
#include "test/test_signing.hpp"
#include <iostream>

int
main()
{
  test_falcon::test_field_ops();
  std::cout << "[test] Falcon prime field arithmetic\n";

  test_falcon::test_ntt<ntt::FALCON512_LOG2N>();
  test_falcon::test_ntt<ntt::FALCON1024_LOG2N>();
  std::cout << "[test] (inverse) Number Theoretic Transform over Z_q\n";

  test_falcon::test_fft<ntt::FALCON512_LOG2N>();
  test_falcon::test_fft<ntt::FALCON1024_LOG2N>();
  std::cout << "[test] (inverse) Fast Fourier Transform over Q\n";

  test_falcon::test_fft_split_merge<2>();
  test_falcon::test_fft_split_merge<3>();
  test_falcon::test_fft_split_merge<4>();
  test_falcon::test_fft_split_merge<5>();
  test_falcon::test_fft_split_merge<6>();
  test_falcon::test_fft_split_merge<7>();
  test_falcon::test_fft_split_merge<8>();
  test_falcon::test_fft_split_merge<9>();
  test_falcon::test_fft_split_merge<10>();
  std::cout << "[test] Splitting and merging of polynomials in FFT form\n";

  test_falcon::test_falcon512_samplerz();
  test_falcon::test_falcon1024_samplerz();
  std::cout << "[test] Sampler over the Integers, using KATs\n";

  test_falcon::test_ntru_gen<512>();
  test_falcon::test_ntru_gen<1024>();
  std::cout << "[test] NTRUGen\n";

  test_falcon::test_encoding_pkey<512>();
  test_falcon::test_encoding_pkey<1024>();
  std::cout << "[test] Encode/ Decode Public Key\n";

  test_falcon::test_encoding_skey<512>();
  test_falcon::test_encoding_skey<1024>();
  std::cout << "[test] Encode/ Decode Secret Key\n";

  test_falcon::test_keygen<512>();
  test_falcon::test_keygen<1024>();
  std::cout << "[test] Falcon KeyGen\n";

  test_falcon::test_ff_sampling<512>(165.736617183, 1.277833697);
  test_falcon::test_ff_sampling<1024>(168.388571447, 1.298280334);
  std::cout << "[test] Fast Fourier Sampling\n";

  test_falcon::test_sig_compression<512>(165.736617183, 1.277833697, 34034726);
  test_falcon::test_sig_compression<1024>(168.388571447, 1.298280334, 70265242);
  test_falcon::test_sig_decompression<512>();
  test_falcon::test_sig_decompression<1024>();
  std::cout << "[test] Signature Compression/ Decompression\n";

  test_falcon::test_keygen_sign_verify<512>();
  test_falcon::test_keygen_sign_verify<1024>();
  std::cout << "[test] Keygen -> Sign -> Verify\n";

  return EXIT_SUCCESS;
}
