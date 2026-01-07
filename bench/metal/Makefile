# TT-Metal paths
TT_METAL := $(HOME)/tt-metal
BUILD_DIR := $(TT_METAL)/build_Release

# Environment for running binaries
RUN_ENV := TT_METAL_HOME=$(TT_METAL) TT_METAL_RUNTIME_ROOT=$(TT_METAL) ARCH_NAME=wormhole_b0

# simple_add (low-level Metal API)
METAL_EXAMPLES := $(TT_METAL)/tt_metal/programming_examples
SIMPLE_ADD_DIR := $(METAL_EXAMPLES)/simple_add

# simple_matmul (TTNN C++ API)
TTNN_EXAMPLES := $(TT_METAL)/ttnn/examples
MATMUL_DIR := $(TTNN_EXAMPLES)/matmul

.PHONY: all sync build run clean
.PHONY: add add-sync add-build add-run
.PHONY: matmul matmul-sync matmul-build matmul-run

# Default: simple_add
all: add

# === simple_add (Metal API) ===
add: add-sync add-build add-run

add-sync:
	cp simple_add.cpp $(SIMPLE_ADD_DIR)/
	cp -r kernels $(SIMPLE_ADD_DIR)/

add-build:
	cd $(BUILD_DIR) && ninja simple_add

add-run:
	$(RUN_ENV) $(TT_METAL)/build_Release/programming_examples/simple_add

# === simple_matmul (TTNN C++ API) ===
matmul: matmul-sync matmul-build matmul-run

matmul-sync:
	mkdir -p $(MATMUL_DIR)
	cp simple_matmul.cpp $(MATMUL_DIR)/matmul.cpp

matmul-build:
	cd $(BUILD_DIR) && ninja example_matmul

matmul-run:
	$(RUN_ENV) $(TT_METAL)/build_Release/ttnn/examples/example_matmul

# === matmul_sweep (TTNN C++ sweep benchmark) ===
sweep: sweep-sync sweep-build sweep-run

sweep-sync:
	mkdir -p $(MATMUL_DIR)
	cp matmul_sweep.cpp $(MATMUL_DIR)/matmul.cpp

sweep-build:
	cd $(BUILD_DIR) && ninja example_matmul

sweep-run:
	$(RUN_ENV) $(TT_METAL)/build_Release/ttnn/examples/example_matmul

# === matmul_2dev (2-device MeshDevice benchmark) ===
# Uses TT_VISIBLE_DEVICES to limit to 2 devices (one N300 card: chips 0 and 4)
2dev: 2dev-sync 2dev-build 2dev-run

2dev-sync:
	mkdir -p $(MATMUL_DIR)
	cp matmul_2dev.cpp $(MATMUL_DIR)/matmul.cpp

2dev-build:
	cd $(BUILD_DIR) && ninja example_matmul

2dev-run:
	$(RUN_ENV) TT_VISIBLE_DEVICES=0,4 $(TT_METAL)/build_Release/ttnn/examples/example_matmul

# === distributed matmul (2-device data parallel - legacy) ===
dist: dist-sync dist-build dist-run

dist-sync:
	mkdir -p $(MATMUL_DIR)
	cp matmul_2dev_distributed.cpp $(MATMUL_DIR)/matmul.cpp

dist-build:
	cd $(BUILD_DIR) && ninja example_matmul

dist-run:
	$(RUN_ENV) TT_VISIBLE_DEVICES=0,4 $(TT_METAL)/build_Release/ttnn/examples/example_matmul

# === tensor parallel matmul (TRUE distributed - column sharding + all_gather) ===
tp: tp-sync tp-build tp-run

tp-sync:
	mkdir -p $(MATMUL_DIR)
	cp matmul_tensor_parallel.cpp $(MATMUL_DIR)/matmul.cpp

tp-build:
	cd $(BUILD_DIR) && ninja example_matmul

tp-run:
	$(RUN_ENV) TT_VISIBLE_DEVICES=0,4 $(TT_METAL)/build_Release/ttnn/examples/example_matmul

# === Command Queue Async vs Sync Benchmark ===
cq: cq-sync cq-build cq-run

cq-sync:
	mkdir -p $(MATMUL_DIR)
	cp bench_cq_async.cpp $(MATMUL_DIR)/matmul.cpp

cq-build:
	cd $(BUILD_DIR) && ninja example_matmul

cq-run:
	$(RUN_ENV) $(TT_METAL)/build_Release/ttnn/examples/example_matmul

# === Memory Config & Fusion Benchmark ===
fusion: fusion-sync fusion-build fusion-run

fusion-sync:
	mkdir -p $(MATMUL_DIR)
	cp bench_fusion.cpp $(MATMUL_DIR)/matmul.cpp

fusion-build:
	cd $(BUILD_DIR) && ninja example_matmul

fusion-run:
	$(RUN_ENV) $(TT_METAL)/build_Release/ttnn/examples/example_matmul

# === Linear-ReLU-Linear CQ Benchmark ===
lrl: lrl-sync lrl-build lrl-run

lrl-sync:
	mkdir -p $(MATMUL_DIR)
	cp bench_linear_relu_linear.cpp $(MATMUL_DIR)/matmul.cpp

lrl-build:
	cd $(BUILD_DIR) && ninja example_matmul

lrl-run:
	$(RUN_ENV) $(TT_METAL)/build_Release/ttnn/examples/example_matmul

# === BFP8 vs BF16 Matmul Benchmark ===
bfp8: bfp8-sync bfp8-build bfp8-run

bfp8-sync:
	mkdir -p $(MATMUL_DIR)
	cp matmul_bfp8.cpp $(MATMUL_DIR)/matmul.cpp

bfp8-build:
	cd $(BUILD_DIR) && ninja example_matmul

bfp8-run:
	$(RUN_ENV) $(TT_METAL)/build_Release/ttnn/examples/example_matmul

# Legacy aliases
sync: add-sync
build: add-build
run: add-run

clean:
	rm -f $(BUILD_DIR)/programming_examples/simple_add
	rm -f $(BUILD_DIR)/ttnn/examples/example_matmul

# === GEMM Top 10 Benchmark ===
top10: top10-sync top10-build top10-run

top10-sync:
	mkdir -p $(MATMUL_DIR)
	cp bench_gemm_top10.cpp $(MATMUL_DIR)/matmul.cpp

top10-build:
	cd $(BUILD_DIR) && ninja example_matmul

top10-run:
	$(RUN_ENV) $(TT_METAL)/build_Release/ttnn/examples/example_matmul

# === GEMM Sharded Benchmark (L1 sharding for smaller matrices) ===
sharded: sharded-sync sharded-build sharded-run

sharded-sync:
	mkdir -p $(MATMUL_DIR)
	cp bench_gemm_sharded.cpp $(MATMUL_DIR)/matmul.cpp

sharded-build:
	cd $(BUILD_DIR) && ninja example_matmul

sharded-run:
	$(RUN_ENV) $(TT_METAL)/build_Release/ttnn/examples/example_matmul

# === GEMM BFP8 Top 10 Benchmark ===
bfp8-top10: bfp8-top10-sync bfp8-top10-build bfp8-top10-run

bfp8-top10-sync:
	mkdir -p $(MATMUL_DIR)
	cp bench_gemm_bfp8.cpp $(MATMUL_DIR)/matmul.cpp

bfp8-top10-build:
	cd $(BUILD_DIR) && ninja example_matmul

bfp8-top10-run:
	$(RUN_ENV) $(TT_METAL)/build_Release/ttnn/examples/example_matmul

# === GEMM BF16 Top 10 Benchmark (HiFi2) ===
bf16-top10: bf16-top10-sync bf16-top10-build bf16-top10-run

bf16-top10-sync:
	mkdir -p $(MATMUL_DIR)
	cp bench_gemm_bf16_top10.cpp $(MATMUL_DIR)/matmul.cpp

bf16-top10-build:
	cd $(BUILD_DIR) && ninja example_matmul

bf16-top10-run:
	$(RUN_ENV) $(TT_METAL)/build_Release/ttnn/examples/example_matmul
