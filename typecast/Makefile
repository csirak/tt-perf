# Typecast Makefile
# Two targets:
#   typecast - TTNN-level benchmark using ttnn::typecast
#   kernel   - Custom Metal kernel for optimization experiments

TT_METAL := $(HOME)/tt-metal
BUILD_DIR := $(TT_METAL)/build_Release
MATMUL_DIR := $(TT_METAL)/ttnn/examples/matmul
# Metal programming_examples - now enabled with BUILD_PROGRAMMING_EXAMPLES=ON
KERNEL_DIR := $(TT_METAL)/tt_metal/programming_examples/eltwise_binary

RUN_ENV := TT_METAL_HOME=$(TT_METAL) TT_METAL_RUNTIME_ROOT=$(TT_METAL) ARCH_NAME=wormhole_b0

.PHONY: all typecast typecast-sync typecast-build typecast-run
.PHONY: kernel kernel-sync kernel-build kernel-run
.PHONY: clean restore-simple-add

all: typecast

# === TTNN Benchmark (uses ttnn::typecast) ===
typecast: typecast-sync typecast-build typecast-run

typecast-sync:
	@echo "=== Syncing bench_typecast.cpp to tt-metal examples ==="
	mkdir -p $(MATMUL_DIR)
	cp bench_typecast.cpp $(MATMUL_DIR)/matmul.cpp

typecast-build:
	@echo "=== Building typecast benchmark ==="
	cd $(BUILD_DIR) && ninja example_matmul

typecast-run:
	@echo "=== Running typecast benchmark ==="
	$(RUN_ENV) $(BUILD_DIR)/ttnn/examples/example_matmul

# === Custom Metal Kernel (BFP8 → BF16) ===
kernel: kernel-sync kernel-build kernel-run

kernel-sync:
	@echo "=== Syncing custom typecast kernel (BFP8 → BF16) ==="
	mkdir -p $(KERNEL_DIR)/kernels/dataflow
	mkdir -p $(KERNEL_DIR)/kernels/compute
	cp typecast_kernel.cpp $(KERNEL_DIR)/eltwise_binary.cpp
	cp kernels/dataflow/reader.cpp $(KERNEL_DIR)/kernels/dataflow/
	cp kernels/dataflow/writer.cpp $(KERNEL_DIR)/kernels/dataflow/
	cp kernels/compute/typecast.cpp $(KERNEL_DIR)/kernels/compute/

kernel-build:
	@echo "=== Building custom typecast kernel ==="
	cd $(BUILD_DIR) && ninja metal_example_eltwise_binary

kernel-run:
	@echo "=== Running custom typecast kernel ==="
	$(RUN_ENV) $(BUILD_DIR)/programming_examples/metal_example_eltwise_binary

# === Custom Metal Kernel (BF16 → BFP8) ===
kernel-bf16-to-bfp8: kernel-bf16-to-bfp8-sync kernel-build kernel-bf16-to-bfp8-run

kernel-bf16-to-bfp8-sync:
	@echo "=== Syncing custom typecast kernel (BF16 → BFP8) ==="
	mkdir -p $(KERNEL_DIR)/kernels/dataflow
	mkdir -p $(KERNEL_DIR)/kernels/compute
	cp typecast_bf16_to_bfp8_kernel.cpp $(KERNEL_DIR)/eltwise_binary.cpp
	cp kernels/dataflow/reader_bf16.cpp $(KERNEL_DIR)/kernels/dataflow/reader_bf16.cpp
	cp kernels/dataflow/writer_bfp8.cpp $(KERNEL_DIR)/kernels/dataflow/writer_bfp8.cpp
	cp kernels/compute/typecast_bf16_to_bfp8.cpp $(KERNEL_DIR)/kernels/compute/

kernel-bf16-to-bfp8-run:
	@echo "=== Running BF16 → BFP8 typecast kernel ==="
	$(RUN_ENV) $(BUILD_DIR)/programming_examples/metal_example_eltwise_binary

# === Benchmark Custom Kernel vs ttnn::typecast ===
bench-kernel: bench-kernel-sync kernel-build bench-kernel-run

bench-kernel-sync:
	@echo "=== Syncing custom kernel benchmark ==="
	mkdir -p $(KERNEL_DIR)/kernels/dataflow
	mkdir -p $(KERNEL_DIR)/kernels/compute
	cp bench_custom_kernel.cpp $(KERNEL_DIR)/eltwise_binary.cpp
	cp kernels/dataflow/reader.cpp $(KERNEL_DIR)/kernels/dataflow/
	cp kernels/dataflow/writer.cpp $(KERNEL_DIR)/kernels/dataflow/
	cp kernels/compute/typecast.cpp $(KERNEL_DIR)/kernels/compute/

bench-kernel-run:
	@echo "=== Running custom kernel benchmark ==="
	$(RUN_ENV) $(BUILD_DIR)/programming_examples/metal_example_eltwise_binary

# === L1-to-L1 Kernel (no DRAM reader/writer) ===
kernel-l1: kernel-l1-sync kernel-build kernel-l1-run

kernel-l1-sync:
	@echo "=== Syncing L1-to-L1 typecast kernel ==="
	mkdir -p $(KERNEL_DIR)/kernels/compute
	cp typecast_l1_kernel.cpp $(KERNEL_DIR)/eltwise_binary.cpp
	cp kernels/compute/typecast_bf16_to_bfp8.cpp $(KERNEL_DIR)/kernels/compute/

kernel-l1-run:
	@echo "=== Running L1-to-L1 typecast kernel ==="
	$(RUN_ENV) $(BUILD_DIR)/programming_examples/metal_example_eltwise_binary

# === L1 LayerNorm + Typecast Test ===
l1-test: l1-test-sync typecast-build l1-test-run

l1-test-sync:
	@echo "=== Syncing L1 LayerNorm + Typecast test ==="
	mkdir -p $(MATMUL_DIR)
	cp test_l1_layernorm_typecast.cpp $(MATMUL_DIR)/matmul.cpp

l1-test-run:
	@echo "=== Running L1 LayerNorm + Typecast test ==="
	$(RUN_ENV) $(BUILD_DIR)/ttnn/examples/example_matmul

# Individual steps
sync: typecast-sync
build: typecast-build
run: typecast-run

# Cleanup
clean:
	rm -f $(BUILD_DIR)/ttnn/examples/example_matmul

restore-simple-add:
	@echo "=== Restoring original simple_add ==="
	cd ~/ttnn-perf/bench/metal && make add-sync
