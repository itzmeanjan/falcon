#include "test/test_falcon.hpp"
#include <iostream>

int
main()
{
  test_falcon::test_field_ops();
  std::cout << "[test] Falcon prime field operations\n";

  return EXIT_SUCCESS;
}
