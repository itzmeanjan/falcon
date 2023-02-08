CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -pedantic
OPTFLAGS = -O3 -march=native -mtune=native
IFLAGS = -I ./include
DEP_IFLAGS = -I ./sha3/include
# From https://gmplib.org/manual/Headers-and-Libraries
LFLAGS = -lgmpxx -lgmp

all: testing

test/a.out: test/main.cpp include/*.hpp include/test/*.hpp
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(IFLAGS) $(DEP_IFLAGS) $< -o $@ $(LFLAGS)

testing: test/a.out
	./$<

bench/a.out: bench/main.cpp include/*.hpp include/bench/*.hpp
	# make sure you've google-benchmark globally installed;
	# see https://github.com/google/benchmark/tree/84c71fa#installation
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(IFLAGS) $(DEP_IFLAGS) $< -o $@ $(LFLAGS) -lbenchmark

benchmark: bench/a.out
    ./$< --benchmark_time_unit=us --benchmark_counters_tabular=true

clean:
	find . -name '*.out' -o -name '*.o' -o -name '*.so' -o -name '*.gch' | xargs rm -rf

format:
	find . -name '*.hpp' -o -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i --style=Mozilla
