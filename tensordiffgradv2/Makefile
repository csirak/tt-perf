# tensordiffgradv2 Makefile

BUILD_DIR := build
TORCH_DIR := $(HOME)/.tenstorrent-venv/lib/python3.10/site-packages/torch

# TT-Metal paths for TTNN builds
TT_METAL := $(HOME)/tt-metal
TT_BUILD_DIR := $(TT_METAL)/build_Release
TTNN_EXAMPLES := $(TT_METAL)/ttnn/examples
TENSORDIFFGRADV2_DIR := $(TTNN_EXAMPLES)/tensordiffgradv2

.PHONY: all build clean test test-torch test-ttnn ttnn-sync ttnn-build ttnn-run

all: build

# ========== Torch backend (standalone cmake) ==========
build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_PREFIX_PATH="$(TORCH_DIR)" \
		-DCMAKE_BUILD_TYPE=Release
	@cd $(BUILD_DIR) && make -j$$(nproc)

debug:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_PREFIX_PATH="$(TORCH_DIR)" \
		-DCMAKE_BUILD_TYPE=Debug
	@cd $(BUILD_DIR) && make -j$$(nproc)

test: test-torch

test-torch: build
	@cd $(BUILD_DIR) && ./torch_ffn_test

# FFN comparison test (Torch only: FP32 + BF16)
test-ffn-compare: build
	@cd $(BUILD_DIR) && ./ffn_compare_torch

clean:
	@rm -rf $(BUILD_DIR)
	@rm -rf outputs/

# ========== TTNN backend (uses tt-metal build system) ==========
# Syncs code to tt-metal/ttnn/examples and builds with ninja

ttnn-setup:
	@mkdir -p $(TENSORDIFFGRADV2_DIR)
	@echo "Created $(TENSORDIFFGRADV2_DIR)"
	@echo "NOTE: Add tensordiffgradv2 to ttnn/examples/CMakeLists.txt:"
	@echo "  add_executable(example_tensordiffgradv2)"
	@echo "  target_sources(example_tensordiffgradv2 PRIVATE tensordiffgradv2/main.cpp)"
	@echo "  target_link_libraries(example_tensordiffgradv2 PRIVATE TTNN::CPP)"

ttnn-sync:
	@mkdir -p $(TENSORDIFFGRADV2_DIR)/core
	@mkdir -p $(TENSORDIFFGRADV2_DIR)/backends
	@mkdir -p $(TENSORDIFFGRADV2_DIR)/autograd
	cp core/dtype.hpp $(TENSORDIFFGRADV2_DIR)/core/
	cp core/tensor.hpp $(TENSORDIFFGRADV2_DIR)/core/
	cp core/op_log.hpp $(TENSORDIFFGRADV2_DIR)/core/
	cp backends/torch_tensor.hpp $(TENSORDIFFGRADV2_DIR)/backends/
	cp backends/ttnn_tensor.hpp $(TENSORDIFFGRADV2_DIR)/backends/
	cp autograd/value.hpp $(TENSORDIFFGRADV2_DIR)/autograd/
	cp autograd/graph.hpp $(TENSORDIFFGRADV2_DIR)/autograd/
	cp tests/ttnn_ffn_test.cpp $(TENSORDIFFGRADV2_DIR)/main.cpp
	@echo "Synced to $(TENSORDIFFGRADV2_DIR)"

# Sync FFN compare test for TTNN build
ttnn-sync-ffn-compare:
	@mkdir -p $(TENSORDIFFGRADV2_DIR)/core
	@mkdir -p $(TENSORDIFFGRADV2_DIR)/backends
	@mkdir -p $(TENSORDIFFGRADV2_DIR)/autograd
	@mkdir -p $(TENSORDIFFGRADV2_DIR)/ops
	cp core/dtype.hpp $(TENSORDIFFGRADV2_DIR)/core/
	cp core/tensor.hpp $(TENSORDIFFGRADV2_DIR)/core/
	cp core/op_log.hpp $(TENSORDIFFGRADV2_DIR)/core/
	cp backends/torch_tensor.hpp $(TENSORDIFFGRADV2_DIR)/backends/
	cp backends/ttnn_tensor.hpp $(TENSORDIFFGRADV2_DIR)/backends/
	cp autograd/value.hpp $(TENSORDIFFGRADV2_DIR)/autograd/
	cp autograd/graph.hpp $(TENSORDIFFGRADV2_DIR)/autograd/
	cp ops/reglu.hpp $(TENSORDIFFGRADV2_DIR)/ops/
	cp tests/ffn_compare_test.cpp $(TENSORDIFFGRADV2_DIR)/main.cpp
	@echo "Synced FFN compare test to $(TENSORDIFFGRADV2_DIR)"

# FFN compare test (TTNN mode)
# Copy outputs dir to tt-metal so TTNN test can find weights
# Note: Torch test outputs to build/outputs, so copy from there
test-ffn-compare-ttnn: ttnn-sync-ffn-compare ttnn-build
	@if [ -d $(BUILD_DIR)/outputs ]; then cp -r $(BUILD_DIR)/outputs $(TT_METAL)/; fi
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_tensordiffgradv2
	@if [ -d $(TT_METAL)/outputs/ffn_bf16_ttnn ]; then \
		cp -r $(TT_METAL)/outputs/ffn_bf16_ttnn $(BUILD_DIR)/outputs/; \
		echo "Copied TTNN outputs to $(BUILD_DIR)/outputs/ffn_bf16_ttnn"; \
	fi

ttnn-build:
	cd $(TT_BUILD_DIR) && ninja example_tensordiffgradv2

ttnn-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_tensordiffgradv2

test-ttnn: ttnn-sync ttnn-build ttnn-run
