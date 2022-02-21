CXX = dpcpp
CXXFLAGS = -Wall -std=c++20
SYCLFLAGS = -fsycl
SYCLCUDAFLAGS = -fsycl-targets=nvptx64-nvidia-cuda
OPTFLAGS = -O3
# see https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compiler-reference/compiler-options/compiler-option-details/floating-point-options/fp-model-fp.html
FPFLAGS = -fp-model=strict
IFLAGS = -I ./include

all: test_falcon

test/a.out: test/main.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(SYCLFLAGS) $(IFLAGS) $< -o $@

test_falcon: test/a.out
	./test/a.out

clean:
	find . -name 'a.out' -o -name '*.o' | xargs rm -f

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i --style=Mozilla
