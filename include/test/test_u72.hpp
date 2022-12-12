#pragma once
#include "u72.hpp"
#include <cassert>

namespace test {

void
u72_ops()
{
  const char* str0 = "3024686241123004913666";
  const char* str1 = "4722366482869645213695";
  const uint8_t bytes0[9] = { 255, 255, 255, 255, 255, 255, 255, 255, 254 };
  const char* str2 = "1";
  const uint8_t bytes1[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 254 };

  const u72::u72 v0 = u72::from_decimal(str0, 22);
  const u72::u72 v1 = u72::from_decimal(str1, 22);
  const u72::u72 v2 = u72::from_bytes(bytes0);
  const u72::u72 v3 = u72::from_decimal(str2, 1);
  const u72::u72 v4 = u72::from_bytes(bytes1);

  assert(u72::cmp(v0, v0) == 0);
  assert(u72::cmp(v1, v1) == 0);
  assert(u72::cmp(v2, v2) == 0);
  assert(u72::cmp(v3, v3) == 0);
  assert(u72::cmp(v0, v1) == -1);
  assert(u72::cmp(v1, v0) == 1);
  assert(u72::cmp(v1, v2) == 1);
  assert(u72::cmp(v2, v1) == -1);
  assert(u72::cmp(v2, v3) == 1);
  assert(u72::cmp(v3, v2) == -1);
  assert(u72::cmp(v2, v4) == 1);
  assert(u72::cmp(v4, v2) == -1);
}

}
