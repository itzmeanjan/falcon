#include "test_ff.hpp"
#include "test_fft.hpp"
#include "test_polynomial.hpp"
#include <iostream>

int
main(int argc, char** argv)
{
  sycl::default_selector s{};
  sycl::device d{ s };
  sycl::context c{ d };
  sycl::queue q{ c, d };

  std::cout << "running on " << d.get_info<sycl::info::device::name>()
            << std::endl
            << std::endl;

  std::cout << "[test] maximum deviation for fft size  512 :\t"
            << test::fft(q, 512, 32) << std::endl;
  std::cout << "[test] maximum deviation for fft size 1024 :\t"
            << test::fft(q, 1024, 32) << std::endl;
  std::cout << "[test] maximum deviation for poly-mul size 512 :\t"
            << test::mul(q, 512, 32) << std::endl;
  std::cout << "[test] maximum deviation for poly-mul size 1024 :\t"
            << test::mul(q, 1024, 32) << std::endl;

  test::ff_math(q);
  std::cout << "[test] passed finite field arithmetic" << std::endl;

  return EXIT_SUCCESS;
}
