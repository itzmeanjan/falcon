CXX ?= clang++
CXX_FLAGS = -std=c++20
WARN_FLAGS = -Wall -Wextra -pedantic
OPT_FLAGS = -O3 -march=native
LINK_FLAGS = -flto
GMP_LINK_FLAGS = -lgmpxx -lgmp

SHA3_INC_DIR = ./sha3/include
I_FLAGS = -I ./include
DEP_IFLAGS = -I $(SHA3_INC_DIR)

SRC_DIR = include
FALCON_SOURCES := $(wildcard $(SRC_DIR)/*.hpp)
BUILD_DIR = build

TEST_DIR = tests
TEST_SOURCES := $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJECTS := $(addprefix $(BUILD_DIR)/, $(notdir $(patsubst %.cpp,%.o,$(TEST_SOURCES))))
TEST_LINK_FLAGS = -lgtest -lgtest_main
TEST_BINARY = $(BUILD_DIR)/test.out
GTEST_PARALLEL = ./gtest-parallel/gtest-parallel

BENCHMARK_DIR = benchmarks
BENCHMARK_SOURCES := $(wildcard $(BENCHMARK_DIR)/*.cpp)
BENCHMARK_HEADERS := $(wildcard $(BENCHMARK_DIR)/*.hpp)
BENCHMARK_OBJECTS := $(addprefix $(BUILD_DIR)/, $(notdir $(patsubst %.cpp,%.o,$(BENCHMARK_SOURCES))))
BENCHMARK_LINK_FLAGS = -lbenchmark -lbenchmark_main -lpthread
BENCHMARK_BINARY = $(BUILD_DIR)/bench.out
PERF_LINK_FLAGS = -lbenchmark -lbenchmark_main -lpfm -lpthread
PERF_BINARY = $(BUILD_DIR)/perf.out

all: test

$(BUILD_DIR):
	mkdir -p $@

$(SHA3_INC_DIR):
	git submodule update --init

$(GTEST_PARALLEL): $(SHA3_INC_DIR)
	git submodule update --init

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp $(BUILD_DIR) $(SHA3_INC_DIR)
	$(CXX) $(CXX_FLAGS) $(WARN_FLAGS) $(OPT_FLAGS) $(I_FLAGS) $(DEP_IFLAGS) -c $< -o $@

$(TEST_BINARY): $(TEST_OBJECTS)
	$(CXX) $(OPT_FLAGS) $(LINK_FLAGS) $^ $(TEST_LINK_FLAGS) -o $@ $(GMP_LINK_FLAGS)

test: $(TEST_BINARY) $(GTEST_PARALLEL)
	$(GTEST_PARALLEL) $< --print_test_times

$(BUILD_DIR)/%.o: $(BENCHMARK_DIR)/%.cpp $(BUILD_DIR) $(SHA3_INC_DIR)
	$(CXX) $(CXX_FLAGS) $(WARN_FLAGS) $(OPT_FLAGS) $(I_FLAGS) $(DEP_IFLAGS) -c $< -o $@

$(BENCHMARK_BINARY): $(BENCHMARK_OBJECTS)
	$(CXX) $(OPT_FLAGS) $(LINK_FLAGS) $^ $(BENCHMARK_LINK_FLAGS) -o $@ $(GMP_LINK_FLAGS)

benchmark: $(BENCHMARK_BINARY)
	# Must *not* build google-benchmark with libPFM
	./$< --benchmark_time_unit=us --benchmark_min_warmup_time=.5 --benchmark_enable_random_interleaving=true --benchmark_repetitions=16 --benchmark_min_time=0.1s --benchmark_display_aggregates_only=true --benchmark_counters_tabular=true

$(PERF_BINARY): $(BENCHMARK_OBJECTS)
	$(CXX) $(OPT_FLAGS) $(LINK_FLAGS) $^ $(PERF_LINK_FLAGS) -o $@ $(GMP_LINK_FLAGS)

perf: $(PERF_BINARY)
	# Must build google-benchmark with libPFM, follow https://gist.github.com/itzmeanjan/05dc3e946f635d00c5e0b21aae6203a7
	./$< --benchmark_time_unit=us --benchmark_min_warmup_time=.5 --benchmark_enable_random_interleaving=true --benchmark_repetitions=16 --benchmark_min_time=0.1s --benchmark_display_aggregates_only=true --benchmark_counters_tabular=true --benchmark_perf_counters=CYCLES

.PHONY: format clean

clean:
	rm -rf $(BUILD_DIR)

format: $(FALCON_SOURCES) $(TEST_SOURCES) $(DUDECT_TEST_SOURCES) $(BENCHMARK_SOURCES) $(BENCHMARK_HEADERS)
	clang-format -i $^
