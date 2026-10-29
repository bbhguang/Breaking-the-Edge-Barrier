CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic
OPENMP_FLAGS ?= -fopenmp
CPPFLAGS ?=
LDFLAGS ?=

.PHONY: all clean
all: build/shuftri_engine

build/shuftri_engine: cpp/engine.cpp
	mkdir -p build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPENMP_FLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f build/shuftri_engine
