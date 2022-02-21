#pragma once
#include <random>

void
random_fill(double* const data, const size_t dim)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<long> dis(-3., 4.);

  for (size_t i = 0; i < dim; i++) {
    data[i] = static_cast<double>(dis(gen));
  }
}
