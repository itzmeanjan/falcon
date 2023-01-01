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

  return EXIT_SUCCESS;
}
