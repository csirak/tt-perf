CXX ?= c++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Wno-unused-parameter

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)

INCLUDES := -I$(ROOT)

OUT_DIR ?= outputs/run0

transformer-test: build/transformer_layer_test
	TENSORDIFFGRAD_OUT_DIR=$(OUT_DIR) ./build/transformer_layer_test

build/transformer_layer_test: tests/transformer_layer_test.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

ffn-test: build/ffn_test
	TENSORDIFFGRAD_OUT_DIR=$(OUT_DIR) ./build/ffn_test

build/ffn_test: tests/ffn_test.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -rf build
