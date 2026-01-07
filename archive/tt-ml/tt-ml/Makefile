# tt-ml Makefile
# Uses tt-train's TTML library for training on TT hardware
# NOTE: tt-ml is built as part of tt-train, not standalone

TT_METAL_HOME ?= $(HOME)/tt-metal
TT_TRAIN_BUILD ?= $(TT_METAL_HOME)/tt-train/build

.PHONY: all build run clean help bench-sweep bench-sweep-build bench-sweep-run bench-sweep-traced bench-sweep-traced-build bench-sweep-traced-run

all: build

build:
	@echo "Building tt-ml as part of tt-train..."
	cd $(TT_METAL_HOME)/tt-train && \
	TT_METAL_HOME=$(TT_METAL_HOME) cmake -B build -GNinja && \
	cmake --build build --target simple_mlp

run: build
	@echo "Running simple_mlp..."
	cd $(TT_TRAIN_BUILD) && \
	TT_METAL_HOME=$(TT_METAL_HOME) \
	TT_METAL_RUNTIME_ROOT=$(TT_METAL_HOME) \
	LD_LIBRARY_PATH=$(TT_METAL_HOME)/build/lib \
	./sources/examples/tt-ml/simple_mlp

clean:
	rm -rf build
	@echo "Note: To fully clean, also run: rm -rf $(TT_TRAIN_BUILD)"

# Scaling sweep benchmark
bench-sweep: bench-sweep-build bench-sweep-run

bench-sweep-build:
	@echo "Building bench_sweep as part of tt-train..."
	cd $(TT_METAL_HOME)/tt-train && \
	TT_METAL_HOME=$(TT_METAL_HOME) cmake -B build -GNinja && \
	cmake --build build --target bench_sweep

bench-sweep-run:
	@echo "Running bench_sweep..."
	cd $(TT_TRAIN_BUILD) && \
	TT_METAL_HOME=$(TT_METAL_HOME) \
	TT_METAL_RUNTIME_ROOT=$(TT_METAL_HOME) \
	LD_LIBRARY_PATH=$(TT_METAL_HOME)/build/lib \
	./sources/examples/tt-ml/bench_sweep

# Scaling sweep benchmark with trace replay
bench-sweep-traced: bench-sweep-traced-build bench-sweep-traced-run

bench-sweep-traced-build:
	@echo "Building bench_sweep_traced as part of tt-train..."
	cd $(TT_METAL_HOME)/tt-train && \
	TT_METAL_HOME=$(TT_METAL_HOME) cmake -B build -GNinja && \
	cmake --build build --target bench_sweep_traced

bench-sweep-traced-run:
	@echo "Running bench_sweep_traced..."
	cd $(TT_TRAIN_BUILD) && \
	TT_METAL_HOME=$(TT_METAL_HOME) \
	TT_METAL_RUNTIME_ROOT=$(TT_METAL_HOME) \
	LD_LIBRARY_PATH=$(TT_METAL_HOME)/build/lib \
	./sources/examples/tt-ml/bench_sweep_traced

# Show help
help:
	@echo "tt-ml - Simple training test with tt-train"
	@echo ""
	@echo "Targets:"
	@echo "  build    - Build simple_mlp via tt-train (default)"
	@echo "  run      - Build and run simple_mlp"
	@echo "  clean    - Remove local build directory"
	@echo ""
	@echo "Environment:"
	@echo "  TT_METAL_HOME    = $(TT_METAL_HOME)"
	@echo "  TT_TRAIN_BUILD   = $(TT_TRAIN_BUILD)"
	@echo ""
	@echo "Prerequisites:"
	@echo "  tt-train added this dir to sources/examples/CMakeLists.txt:"
	@echo "    add_subdirectory(/home/howard/ttnn-perf/tt-ml tt-ml)"
