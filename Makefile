CXX = dpcpp
CXXFLAGS = -Wall -std=c++20
SYCLFLAGS = -fsycl
SYCLCUDAFLAGS = -fsycl-targets=nvptx64-nvidia-cuda
OPTFLAGS = -O3
IFLAGS = -I ./include

all: test_falcon

test/a.out: test/main.cpp include/*.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(IFLAGS) $< -o $@

test_falcon: test/a.out
	./test/a.out

clean:
	find . -name 'a.out' -o -name '*.o' | xargs rm -f

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i --style=Mozilla
