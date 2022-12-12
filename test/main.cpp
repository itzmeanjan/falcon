#include "test/test_falcon.hpp"
#include <iostream>

int
main()
{
  test::ff_math();
  std::cout << "[test] Falcon prime field operations\n";

  return EXIT_SUCCESS;
}
