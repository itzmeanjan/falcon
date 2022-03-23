#include "test_ff.hpp"
#include "test_fft.hpp"
#include "test_karatsuba.hpp"
#include "test_ntru_gen.hpp"
#include "test_ntt.hpp"
#include "test_polynomial.hpp"
#include "test_samplerz.hpp"
#include "test_u72.hpp"
#include "test_utils.hpp"
#include <iomanip>
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

  std::cout << "[test] maximum deviation for poly-mul over R of size  512 :\t"
            << test::mul(q, 512, 32) << std::endl;
  std::cout << "[test] maximum deviation for poly-mul over R of size 1024 :\t"
            << test::mul(q, 1024, 32) << std::endl;

  test::ff_math(q);
  std::cout << "[test] passed prime ( = 12289 ) field arithmetic test"
            << std::endl;

  test::ntt(q, 512, 32);
  std::cout << "[test] passed ntt test for size  512" << std::endl;

  test::ntt(q, 1024, 32);
  std::cout << "[test] passed ntt test for size 1024" << std::endl;

  test::mul_zq(q, 512, 32);
  std::cout << "[test] passed poly-mul test over Z_q of size  512" << std::endl;
  test::mul_zq(q, 1024, 32);
  std::cout << "[test] passed poly-mul test over Z_q of size 1024" << std::endl;

  test::karatsuba(q);
  std::cout << "[test] passed karatsuba polynomial multiplication" << std::endl;

  test::u72_ops();
  std::cout << "[test] passed u72 ops" << std::endl;

  test::samplerz();
  std::cout << "[test] passed samplerz test" << std::endl;

  test::sqnorm(q, 512, 32);
  test::sqnorm(q, 1024, 32);
  std::cout << "[test] passed polynomial coefficient norm compute" << std::endl;

  size_t itr_cnt = 0;
  const double gsn_0 = test::gs_norm(q, 512, 32, &itr_cnt);
  std::cout << "[test] squared gram-schmidt norm of NTRU matrix : " << gsn_0
            << std::setw(16) << std::right << "[ " << itr_cnt << " iterations ]"
            << std::endl;
  const double gsn_1 = test::gs_norm(q, 1024, 32, &itr_cnt);
  std::cout << "[test] squared gram-schmidt norm of NTRU matrix : " << gsn_1
            << std::setw(16) << std::right << "[ " << itr_cnt << " iterations ]"
            << std::endl;

  test::is_nonzero_coeff(q, 1024, 32);
  std::cout << "[test] passed non-zero polynomial coefficient test" << std::endl;

  return EXIT_SUCCESS;
}
