#include "ntt.hpp"
#include "test/test_falcon.hpp"
#include "test/test_ntt.hpp"
#include <iostream>

int
main()
{
  test_falcon::test_field_ops();
  std::cout << "[test] Falcon prime field operations\n";

  test_falcon::test_ntt<ntt::FALCON512_LOG2N>();
  test_falcon::test_ntt<ntt::FALCON1024_LOG2N>();
  std::cout << "[test] (inverse) Number Theoretic Transform over field Z_q\n";

  return EXIT_SUCCESS;
}
