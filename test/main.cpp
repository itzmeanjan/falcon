#include "test/test_falcon.hpp"
#include <iostream>

int
main()
{
  test_falcon::test_field_ops();
  std::cout << "[test] Falcon prime field arithmetic\n";

  test_falcon::test_ntt<ntt::FALCON512_LOG2N>();
  test_falcon::test_ntt<ntt::FALCON1024_LOG2N>();
  std::cout << "[test] (inverse) Number Theoretic Transform over field Z_q\n";

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

  return EXIT_SUCCESS;
}
