# C++ Autograd on TTNN
# Uses the tt-metal build system by copying files to example directories

TT_METAL := ~/tt-metal
BUILD_DIR := $(TT_METAL)/build_Release
TTNN_EXAMPLES := $(TT_METAL)/ttnn/examples
AUTOGRAD_DIR := $(TTNN_EXAMPLES)/autograd

.PHONY: all clean setup
.PHONY: tensor matmul relu softmax linear sgd train mlp mlp-traced
.PHONY: core-grid core-grid-profile
.PHONY: verify verify-sync verify-build verify-run verify-torch
.PHONY: fidelity fidelity-sync fidelity-build fidelity-run
.PHONY: traced-mlp traced-mlp-sync traced-mlp-build traced-mlp-run
.PHONY: bench-mlp bench-mlp-sync bench-mlp-build bench-mlp-run
.PHONY: bench-autograd bench-autograd-sync bench-autograd-build bench-autograd-run
.PHONY: attention attention-sync attention-build attention-run
.PHONY: static-autograd static-autograd-sync static-autograd-build static-autograd-run
.PHONY: bench-static-trace bench-static-trace-sync bench-static-trace-build bench-static-trace-run bench-static-trace-profile
.PHONY: bench-traced-trace bench-traced-trace-sync bench-traced-trace-build bench-traced-trace-run bench-traced-trace-profile
.PHONY: bench-fusion bench-fusion-sync bench-fusion-build bench-fusion-run
.PHONY: bench-fusion-fwd bench-fusion-fwd-sync bench-fusion-fwd-build bench-fusion-fwd-run
.PHONY: bench-core-grid bench-core-grid-sync bench-core-grid-build bench-core-grid-run
.PHONY: bench-congestion bench-congestion-sync bench-congestion-build bench-congestion-run
.PHONY: bench-multicast bench-multicast-sync bench-multicast-build bench-multicast-run
.PHONY: bench-parallel-subdevice bench-parallel-subdevice-sync bench-parallel-subdevice-build bench-parallel-subdevice-run
.PHONY: bench-minimal bench-minimal-sync bench-minimal-build bench-minimal-run bench-minimal-noc
.PHONY: bench-fullgrid bench-fullgrid-sync bench-fullgrid-build bench-fullgrid-run bench-fullgrid-noc
.PHONY: bench-partitioned bench-partitioned-sync bench-partitioned-build bench-partitioned-run bench-partitioned-noc
.PHONY: bench-7x2x4 bench-7x2x4-sync bench-7x2x4-build bench-7x2x4-run bench-7x2x4-noc
.PHONY: bench-fullgrid-small bench-fullgrid-small-sync bench-fullgrid-small-build bench-fullgrid-small-run
.PHONY: bench-4x2x4-uniform bench-4x2x4-uniform-sync bench-4x2x4-uniform-build bench-4x2x4-uniform-run bench-4x2x4-uniform-noc
.PHONY: bench-7x2x4-uniform bench-7x2x4-uniform-sync bench-7x2x4-uniform-build bench-7x2x4-uniform-run bench-7x2x4-uniform-noc
.PHONY: bench-matmul-compare bench-matmul-compare-sync bench-matmul-compare-build bench-matmul-compare-run bench-matmul-compare-noc bench-matmul-compare-profile
.PHONY: bench-matmul-parallel-cq bench-matmul-parallel-cq-sync bench-matmul-parallel-cq-build bench-matmul-parallel-cq-run
.PHONY: bench-matmul-traced bench-matmul-traced-sync bench-matmul-traced-build bench-matmul-traced-run
.PHONY: bench-partition-sweep bench-partition-sweep-sync bench-partition-sweep-build bench-partition-sweep-run
.PHONY: bench-transformer-depth bench-transformer-depth-sync bench-transformer-depth-build bench-transformer-depth-run
.PHONY: bench-gpt2-small bench-gpt2-small-sync bench-gpt2-small-build bench-gpt2-small-run bench-gpt2-small-noc
.PHONY: mesh mesh-sync mesh-build mesh-run

# Default target
all: tensor

# Create example directory (one-time setup)
setup:
	mkdir -p $(AUTOGRAD_DIR)
	@echo "Created $(AUTOGRAD_DIR)"
	@echo "You may need to add autograd to CMakeLists.txt in ttnn/examples/"

# === Test: Basic tensor wrapper ===
tensor: tensor-sync tensor-build tensor-run

tensor-sync:
	mkdir -p $(AUTOGRAD_DIR)/dynamic
	cp dynamic/autograd.hpp $(AUTOGRAD_DIR)/dynamic/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp tests/test_tensor.cpp $(AUTOGRAD_DIR)/main.cpp

tensor-build:
	cd $(BUILD_DIR) && ninja example_autograd

tensor-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Test: Matmul forward/backward ===
matmul: matmul-sync matmul-build matmul-run

matmul-sync:
	mkdir -p $(AUTOGRAD_DIR)/dynamic
	cp dynamic/autograd.hpp $(AUTOGRAD_DIR)/dynamic/
	cp dynamic/ops.hpp $(AUTOGRAD_DIR)/dynamic/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp tests/test_matmul.cpp $(AUTOGRAD_DIR)/main.cpp

matmul-build:
	cd $(BUILD_DIR) && ninja example_autograd

matmul-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Test: ReLU ===
relu: relu-sync relu-build relu-run

relu-sync:
	mkdir -p $(AUTOGRAD_DIR)/dynamic
	cp dynamic/autograd.hpp $(AUTOGRAD_DIR)/dynamic/
	cp dynamic/ops.hpp $(AUTOGRAD_DIR)/dynamic/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp tests/test_relu.cpp $(AUTOGRAD_DIR)/main.cpp

relu-build:
	cd $(BUILD_DIR) && ninja example_autograd

relu-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Test: Softmax + CrossEntropy ===
softmax: softmax-sync softmax-build softmax-run

softmax-sync:
	mkdir -p $(AUTOGRAD_DIR)/dynamic
	cp dynamic/autograd.hpp $(AUTOGRAD_DIR)/dynamic/
	cp dynamic/ops.hpp $(AUTOGRAD_DIR)/dynamic/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp tests/test_softmax.cpp $(AUTOGRAD_DIR)/main.cpp

softmax-build:
	cd $(BUILD_DIR) && ninja example_autograd

softmax-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Test: Linear layer ===
linear: linear-sync linear-build linear-run

linear-sync:
	mkdir -p $(AUTOGRAD_DIR)/dynamic
	cp dynamic/autograd.hpp $(AUTOGRAD_DIR)/dynamic/
	cp dynamic/ops.hpp $(AUTOGRAD_DIR)/dynamic/
	cp dynamic/nn.hpp $(AUTOGRAD_DIR)/dynamic/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp tests/test_linear.cpp $(AUTOGRAD_DIR)/main.cpp

linear-build:
	cd $(BUILD_DIR) && ninja example_autograd

linear-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Test: SGD Optimizer ===
sgd: sgd-sync sgd-build sgd-run

sgd-sync:
	mkdir -p $(AUTOGRAD_DIR)/dynamic
	cp dynamic/autograd.hpp $(AUTOGRAD_DIR)/dynamic/
	cp dynamic/ops.hpp $(AUTOGRAD_DIR)/dynamic/
	cp dynamic/nn.hpp $(AUTOGRAD_DIR)/dynamic/
	cp dynamic/optim.hpp $(AUTOGRAD_DIR)/dynamic/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp tests/test_sgd.cpp $(AUTOGRAD_DIR)/main.cpp

sgd-build:
	cd $(BUILD_DIR) && ninja example_autograd

sgd-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Final: Train FFN ===
train: train-sync train-build train-run

train-sync:
	mkdir -p $(AUTOGRAD_DIR)/dynamic
	cp dynamic/autograd.hpp $(AUTOGRAD_DIR)/dynamic/
	cp dynamic/ops.hpp $(AUTOGRAD_DIR)/dynamic/
	cp dynamic/nn.hpp $(AUTOGRAD_DIR)/dynamic/
	cp dynamic/optim.hpp $(AUTOGRAD_DIR)/dynamic/
	cp train_ffn.cpp $(AUTOGRAD_DIR)/main.cpp

train-build:
	cd $(BUILD_DIR) && ninja example_autograd

train-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === MLP Training Experiment (Dynamic) ===
mlp: mlp-sync mlp-build mlp-run

mlp-sync:
	mkdir -p $(AUTOGRAD_DIR)/dynamic
	cp dynamic/autograd.hpp $(AUTOGRAD_DIR)/dynamic/
	cp dynamic/ops.hpp $(AUTOGRAD_DIR)/dynamic/
	cp dynamic/nn.hpp $(AUTOGRAD_DIR)/dynamic/
	cp dynamic/optim.hpp $(AUTOGRAD_DIR)/dynamic/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../examples/train_mlp.cpp $(AUTOGRAD_DIR)/main.cpp

mlp-build:
	cd $(BUILD_DIR) && ninja example_autograd

mlp-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === MLP Training with Trace API (Traced) ===
traced-mlp: traced-mlp-sync traced-mlp-build traced-mlp-run

traced-mlp-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../examples/train_mlp_traced.cpp $(AUTOGRAD_DIR)/main.cpp

traced-mlp-build:
	cd $(BUILD_DIR) && ninja example_autograd

traced-mlp-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# PyTorch comparison
mlp-torch:
	python3 train_mlp_torch.py

# === Tracy Profiling ===
PROFILER_DIR := $(TT_METAL)/generated/profiler
PROFILER_LOGS := $(PROFILER_DIR)/.logs
CAPTURE_BIN := $(BUILD_DIR)/tools/profiler/bin/capture-release
CSVEXPORT_BIN := $(BUILD_DIR)/tools/profiler/bin/csvexport-release
TRACY_PORT := 8086

mlp-profile: mlp-sync mlp-build
	@echo "=== Tracy Profiling: MLP Training ==="
	@mkdir -p $(PROFILER_LOGS)
	@echo "Starting Tracy capture on port $(TRACY_PORT)..."
	@$(CAPTURE_BIN) -o $(PROFILER_LOGS)/tracy_profile_log_host.tracy -f -p $(TRACY_PORT) & \
	CAPTURE_PID=$$!; \
	sleep 2; \
	echo "Running MLP training with profiling..."; \
	cd $(TT_METAL) && TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_SYNC=1 TRACY_PORT=$(TRACY_PORT) \
		./build_Release/ttnn/examples/example_autograd; \
	sleep 1; \
	kill $$CAPTURE_PID 2>/dev/null || true; \
	wait $$CAPTURE_PID 2>/dev/null || true
	@echo "Exporting to CSV..."
	@$(CSVEXPORT_BIN) -u $(PROFILER_LOGS)/tracy_profile_log_host.tracy \
		> $(PROFILER_LOGS)/tracy_ops_times.csv 2>/dev/null || true
	@echo ""
	@echo "=== Profiling Complete ==="
	@echo "Tracy trace: $(PROFILER_LOGS)/tracy_profile_log_host.tracy"
	@echo "CSV output:  $(PROFILER_LOGS)/tracy_ops_times.csv"
	@echo "Device logs: $(PROFILER_LOGS)/"

# === Core Grid Test ===
core-grid: core-grid-sync core-grid-build core-grid-run

core-grid-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp tests/test_core_grid.cpp $(AUTOGRAD_DIR)/main.cpp

core-grid-build:
	cd $(BUILD_DIR) && ninja example_autograd

core-grid-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

core-grid-profile: core-grid-sync core-grid-build
	@echo "=== Tracy Profiling: Core Grid Test ==="
	@mkdir -p $(PROFILER_LOGS)
	@echo "Starting Tracy capture on port $(TRACY_PORT)..."
	@$(CAPTURE_BIN) -o $(PROFILER_LOGS)/tracy_profile_log_host.tracy -f -p $(TRACY_PORT) & \
	CAPTURE_PID=$$!; \
	sleep 2; \
	echo "Running core grid test with profiling..."; \
	cd $(TT_METAL) && TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_SYNC=1 TRACY_PORT=$(TRACY_PORT) \
		./build_Release/ttnn/examples/example_autograd; \
	sleep 1; \
	kill $$CAPTURE_PID 2>/dev/null || true; \
	wait $$CAPTURE_PID 2>/dev/null || true
	@echo "Exporting to CSV..."
	@$(CSVEXPORT_BIN) -u $(PROFILER_LOGS)/tracy_profile_log_host.tracy \
		> $(PROFILER_LOGS)/tracy_ops_times.csv 2>/dev/null || true
	@echo ""
	@echo "=== Profiling Complete ==="
	@echo "Tracy trace: $(PROFILER_LOGS)/tracy_profile_log_host.tracy"

# === Verification: Compare TTNN vs PyTorch ===
verify: verify-torch verify-sync verify-build verify-run

verify-torch:
	@echo "=== PyTorch Reference ==="
	python3 ../verify/verify_mlp.py

verify-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../verify/verify_mlp_traced.cpp $(AUTOGRAD_DIR)/main.cpp

verify-build:
	cd $(BUILD_DIR) && ninja example_autograd

verify-run:
	@echo ""
	@echo "=== TTNN C++ ==="
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Math Fidelity Comparison ===
fidelity: fidelity-sync fidelity-build fidelity-run

fidelity-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../verify/verify_fidelity.cpp $(AUTOGRAD_DIR)/main.cpp

fidelity-build:
	cd $(BUILD_DIR) && ninja example_autograd

fidelity-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === MLP Benchmark (layers x dims) ===
bench-mlp: bench-mlp-sync bench-mlp-build bench-mlp-run

bench-mlp-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_mlp.cpp $(AUTOGRAD_DIR)/main.cpp

bench-mlp-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-mlp-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Autograd Benchmark (dynamic vs static vs traced) ===
bench-autograd: bench-autograd-sync bench-autograd-build bench-autograd-run

bench-autograd-sync:
	mkdir -p $(AUTOGRAD_DIR)/dynamic
	mkdir -p $(AUTOGRAD_DIR)/static
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r dynamic/* $(AUTOGRAD_DIR)/dynamic/
	cp -r static/* $(AUTOGRAD_DIR)/static/
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_autograd.cpp $(AUTOGRAD_DIR)/main.cpp

bench-autograd-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-autograd-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Test: Causal Attention ===
attention: attention-sync attention-build attention-run

attention-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp tests/test_attention.cpp $(AUTOGRAD_DIR)/main.cpp

attention-build:
	cd $(BUILD_DIR) && ninja example_autograd

attention-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Test: Static Autograd ===
static-autograd: static-autograd-sync static-autograd-build static-autograd-run

static-autograd-sync:
	mkdir -p $(AUTOGRAD_DIR)/static
	cp -r static/* $(AUTOGRAD_DIR)/static/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp tests/test_static.cpp $(AUTOGRAD_DIR)/main.cpp

static-autograd-build:
	cd $(BUILD_DIR) && ninja example_autograd

static-autograd-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Isolated Benchmark: static+trace (for Tracy profiling) ===
bench-static-trace: bench-static-trace-sync bench-static-trace-build bench-static-trace-run

bench-static-trace-sync:
	mkdir -p $(AUTOGRAD_DIR)/static
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r static/* $(AUTOGRAD_DIR)/static/
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_static_trace_only.cpp $(AUTOGRAD_DIR)/main.cpp

bench-static-trace-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-static-trace-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

bench-static-trace-profile: bench-static-trace-sync bench-static-trace-build
	@echo "=== Tracy Profiling: static+trace ==="
	@mkdir -p $(PROFILER_LOGS)
	@echo "Starting Tracy capture on port $(TRACY_PORT)..."
	@$(CAPTURE_BIN) -o $(PROFILER_LOGS)/static_trace.tracy -f -p $(TRACY_PORT) & \
	CAPTURE_PID=$$!; \
	sleep 2; \
	echo "Running static+trace benchmark..."; \
	cd $(TT_METAL) && TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_SYNC=1 TRACY_PORT=$(TRACY_PORT) \
		./build_Release/ttnn/examples/example_autograd; \
	sleep 1; \
	kill $$CAPTURE_PID 2>/dev/null || true; \
	wait $$CAPTURE_PID 2>/dev/null || true
	@echo "Exporting to CSV..."
	@$(CSVEXPORT_BIN) -u $(PROFILER_LOGS)/static_trace.tracy \
		> $(PROFILER_LOGS)/static_trace_ops.csv 2>/dev/null || true
	@echo ""
	@echo "=== Profiling Complete ==="
	@echo "Tracy trace: $(PROFILER_LOGS)/static_trace.tracy"
	@echo "CSV output:  $(PROFILER_LOGS)/static_trace_ops.csv"

# === Isolated Benchmark: traced+trace_api (for Tracy profiling) ===
bench-traced-trace: bench-traced-trace-sync bench-traced-trace-build bench-traced-trace-run

bench-traced-trace-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_traced_only.cpp $(AUTOGRAD_DIR)/main.cpp

bench-traced-trace-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-traced-trace-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

bench-traced-trace-profile: bench-traced-trace-sync bench-traced-trace-build
	@echo "=== Tracy Profiling: traced+trace_api ==="
	@mkdir -p $(PROFILER_LOGS)
	@echo "Starting Tracy capture on port $(TRACY_PORT)..."
	@$(CAPTURE_BIN) -o $(PROFILER_LOGS)/traced_trace.tracy -f -p $(TRACY_PORT) & \
	CAPTURE_PID=$$!; \
	sleep 2; \
	echo "Running traced+trace_api benchmark..."; \
	cd $(TT_METAL) && TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_SYNC=1 TRACY_PORT=$(TRACY_PORT) \
		./build_Release/ttnn/examples/example_autograd; \
	sleep 1; \
	kill $$CAPTURE_PID 2>/dev/null || true; \
	wait $$CAPTURE_PID 2>/dev/null || true
	@echo "Exporting to CSV..."
	@$(CSVEXPORT_BIN) -u $(PROFILER_LOGS)/traced_trace.tracy \
		> $(PROFILER_LOGS)/traced_trace_ops.csv 2>/dev/null || true
	@echo ""
	@echo "=== Profiling Complete ==="
	@echo "Tracy trace: $(PROFILER_LOGS)/traced_trace.tracy"
	@echo "CSV output:  $(PROFILER_LOGS)/traced_trace_ops.csv"

# === Fusion Benchmark: MLP vs FusedMLP ===
bench-fusion: bench-fusion-sync bench-fusion-build bench-fusion-run

bench-fusion-sync:
	mkdir -p $(AUTOGRAD_DIR)/static
	cp -r static/* $(AUTOGRAD_DIR)/static/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_fusion.cpp $(AUTOGRAD_DIR)/main.cpp

bench-fusion-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-fusion-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Fusion Forward-Only Benchmark ===
bench-fusion-fwd: bench-fusion-fwd-sync bench-fusion-fwd-build bench-fusion-fwd-run

bench-fusion-fwd-sync:
	mkdir -p $(AUTOGRAD_DIR)/static
	cp -r static/* $(AUTOGRAD_DIR)/static/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_fusion_forward.cpp $(AUTOGRAD_DIR)/main.cpp

bench-fusion-fwd-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-fusion-fwd-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Core Grid Benchmark: MLP with restricted core grids ===
bench-core-grid: bench-core-grid-sync bench-core-grid-build bench-core-grid-run

bench-core-grid-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_core_grid.cpp $(AUTOGRAD_DIR)/main.cpp

bench-core-grid-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-core-grid-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Congestion Benchmark: Multiple parallel MLPs ===
bench-congestion: bench-congestion-sync bench-congestion-build bench-congestion-run

bench-congestion-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_congestion.cpp $(AUTOGRAD_DIR)/main.cpp

bench-congestion-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-congestion-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Multicast Benchmark: Shared weights vs naive parallel ===
bench-multicast: bench-multicast-sync bench-multicast-build bench-multicast-run

bench-multicast-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_multicast.cpp $(AUTOGRAD_DIR)/main.cpp

bench-multicast-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-multicast-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Parallel SubDevice Benchmark: True data parallel with SubDeviceManager ===
bench-parallel-subdevice: bench-parallel-subdevice-sync bench-parallel-subdevice-build bench-parallel-subdevice-run

bench-parallel-subdevice-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_parallel_subdevice.cpp $(AUTOGRAD_DIR)/main.cpp

bench-parallel-subdevice-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-parallel-subdevice-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Minimal Benchmark: Small trace count for tt-npe analysis ===
bench-minimal: bench-minimal-sync bench-minimal-build bench-minimal-run

bench-minimal-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_minimal.cpp $(AUTOGRAD_DIR)/main.cpp

bench-minimal-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-minimal-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

bench-minimal-noc:
	rm -rf ~/noc_traces_minimal
	mkdir -p ~/noc_traces_minimal
	cd $(TT_METAL) && \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=~/noc_traces_minimal \
		./build_Release/ttnn/examples/example_autograd
	mv ~/noc_traces_minimal/cluster_coordinates.json ~/cluster_coordinates.json.bak 2>/dev/null || true
	@echo "Traces in ~/noc_traces_minimal/"

# === Full Grid Benchmark: No CoreGrid constraint (baseline) ===
bench-fullgrid: bench-fullgrid-sync bench-fullgrid-build bench-fullgrid-run

bench-fullgrid-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_fullgrid.cpp $(AUTOGRAD_DIR)/main.cpp

bench-fullgrid-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-fullgrid-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

bench-fullgrid-noc:
	rm -rf ~/noc_traces_fullgrid
	mkdir -p ~/noc_traces_fullgrid
	cd $(TT_METAL) && \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=~/noc_traces_fullgrid \
		./build_Release/ttnn/examples/example_autograd
	mv ~/noc_traces_fullgrid/cluster_coordinates.json ~/cluster_coordinates.json.bak 2>/dev/null || true
	@echo "Traces in ~/noc_traces_fullgrid/"

# === Fully Partitioned Benchmark: ALL ops constrained to 4x4 cores ===
bench-partitioned: bench-partitioned-sync bench-partitioned-build bench-partitioned-run

bench-partitioned-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_partitioned.cpp $(AUTOGRAD_DIR)/main.cpp

bench-partitioned-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-partitioned-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

bench-partitioned-noc:
	rm -rf ~/noc_traces_partitioned
	mkdir -p ~/noc_traces_partitioned
	cd $(TT_METAL) && \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=~/noc_traces_partitioned \
		./build_Release/ttnn/examples/example_autograd
	mv ~/noc_traces_partitioned/cluster_coordinates.json ~/cluster_coordinates.json.bak 2>/dev/null || true
	@echo "Traces in ~/noc_traces_partitioned/"

# === Benchmark: 7x Parallel 2x4 Core Groups ===
bench-7x2x4: bench-7x2x4-sync bench-7x2x4-build bench-7x2x4-run

bench-7x2x4-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_7x2x4.cpp $(AUTOGRAD_DIR)/main.cpp

bench-7x2x4-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-7x2x4-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

bench-7x2x4-noc:
	rm -rf ~/noc_traces_7x2x4
	mkdir -p ~/noc_traces_7x2x4
	cd $(TT_METAL) && \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=~/noc_traces_7x2x4 \
		./build_Release/ttnn/examples/example_autograd
	mv ~/noc_traces_7x2x4/cluster_coordinates.json ~/cluster_coordinates.json.bak 2>/dev/null || true
	@echo "Traces in ~/noc_traces_7x2x4/"

# === Benchmark: Full Grid Small (for comparison) ===
bench-fullgrid-small: bench-fullgrid-small-sync bench-fullgrid-small-build bench-fullgrid-small-run

bench-fullgrid-small-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_fullgrid_small.cpp $(AUTOGRAD_DIR)/main.cpp

bench-fullgrid-small-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-fullgrid-small-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Benchmark: 4x Parallel 2x4 Uniform Core Groups ===
bench-4x2x4-uniform: bench-4x2x4-uniform-sync bench-4x2x4-uniform-build bench-4x2x4-uniform-run

bench-4x2x4-uniform-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_4x2x4_uniform.cpp $(AUTOGRAD_DIR)/main.cpp

bench-4x2x4-uniform-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-4x2x4-uniform-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

bench-4x2x4-uniform-noc:
	rm -rf ~/noc_traces_4x2x4_uniform
	mkdir -p ~/noc_traces_4x2x4_uniform
	cd $(TT_METAL) && \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=~/noc_traces_4x2x4_uniform \
		./build_Release/ttnn/examples/example_autograd
	mv ~/noc_traces_4x2x4_uniform/cluster_coordinates.json ~/cluster_coordinates.json.bak 2>/dev/null || true
	@echo "Traces in ~/noc_traces_4x2x4_uniform/"

# === Benchmark: 7x Parallel Uniform CoreGrid ===
bench-7x2x4-uniform: bench-7x2x4-uniform-sync bench-7x2x4-uniform-build bench-7x2x4-uniform-run

bench-7x2x4-uniform-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_7x2x4_uniform.cpp $(AUTOGRAD_DIR)/main.cpp

bench-7x2x4-uniform-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-7x2x4-uniform-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

bench-7x2x4-uniform-noc:
	rm -rf ~/noc_traces_7x2x4_uniform
	mkdir -p ~/noc_traces_7x2x4_uniform
	cd $(TT_METAL) && \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=~/noc_traces_7x2x4_uniform \
		./build_Release/ttnn/examples/example_autograd
	mv ~/noc_traces_7x2x4_uniform/cluster_coordinates.json ~/cluster_coordinates.json.bak 2>/dev/null || true
	@echo "Traces in ~/noc_traces_7x2x4_uniform/"

# === Matmul Comparison: Full Grid vs 7x Partitioned ===
bench-matmul-compare: bench-matmul-compare-sync bench-matmul-compare-build bench-matmul-compare-run

bench-matmul-compare-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_matmul_compare.cpp $(AUTOGRAD_DIR)/main.cpp

bench-matmul-compare-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-matmul-compare-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

bench-matmul-compare-noc:
	rm -rf ~/noc_traces_matmul_compare
	mkdir -p ~/noc_traces_matmul_compare
	cd $(TT_METAL) && \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=~/noc_traces_matmul_compare \
		./build_Release/ttnn/examples/example_autograd
	mv ~/noc_traces_matmul_compare/cluster_coordinates.json ~/cluster_coordinates.json.bak 2>/dev/null || true
	@echo "Traces in ~/noc_traces_matmul_compare/"

bench-matmul-compare-profile: bench-matmul-compare-sync bench-matmul-compare-build
	@echo "=== Tracy Profiling: Matmul Compare ==="
	@mkdir -p $(PROFILER_LOGS)
	@echo "Starting Tracy capture on port $(TRACY_PORT)..."
	@$(CAPTURE_BIN) -o $(PROFILER_LOGS)/matmul_compare.tracy -f -p $(TRACY_PORT) & \
	CAPTURE_PID=$$!; \
	sleep 2; \
	echo "Running matmul compare benchmark..."; \
	cd $(TT_METAL) && TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_SYNC=1 TRACY_PORT=$(TRACY_PORT) \
		./build_Release/ttnn/examples/example_autograd; \
	sleep 1; \
	kill $$CAPTURE_PID 2>/dev/null || true; \
	wait $$CAPTURE_PID 2>/dev/null || true
	@echo "Exporting to CSV..."
	@$(CSVEXPORT_BIN) -u $(PROFILER_LOGS)/matmul_compare.tracy \
		> $(PROFILER_LOGS)/matmul_compare_ops.csv 2>/dev/null || true
	@echo ""
	@echo "=== Profiling Complete ==="
	@echo "Tracy trace: $(PROFILER_LOGS)/matmul_compare.tracy"
	@echo "CSV output:  $(PROFILER_LOGS)/matmul_compare_ops.csv"

# === Parallel Matmul with Multiple Command Queues ===
bench-matmul-parallel-cq: bench-matmul-parallel-cq-sync bench-matmul-parallel-cq-build bench-matmul-parallel-cq-run

bench-matmul-parallel-cq-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_matmul_parallel_cq.cpp $(AUTOGRAD_DIR)/main.cpp

bench-matmul-parallel-cq-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-matmul-parallel-cq-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Traced Matmul Comparison: Full Grid vs 7x Partitioned with Trace API ===
bench-matmul-traced: bench-matmul-traced-sync bench-matmul-traced-build bench-matmul-traced-run

bench-matmul-traced-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_matmul_traced.cpp $(AUTOGRAD_DIR)/main.cpp

bench-matmul-traced-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-matmul-traced-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === TracedMeshModel Test: Data-parallel model partitioning (traced) ===
mesh: mesh-sync mesh-build mesh-run

mesh-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp traced/trace.hpp $(AUTOGRAD_DIR)/traced/
	cp traced/ops.hpp $(AUTOGRAD_DIR)/traced/
	cp traced/nn.hpp $(AUTOGRAD_DIR)/traced/
	cp traced/mlp.hpp $(AUTOGRAD_DIR)/traced/
	cp traced/mesh.hpp $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp tests/test_mesh.cpp $(AUTOGRAD_DIR)/main.cpp

mesh-build:
	cd $(BUILD_DIR) && ninja example_autograd

mesh-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Mesh + Trace Benchmark: Partitioned vs Full Grid with Trace API ===
bench-mesh-traced: bench-mesh-traced-sync bench-mesh-traced-build bench-mesh-traced-run

bench-mesh-traced-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp traced/trace.hpp $(AUTOGRAD_DIR)/traced/
	cp traced/ops.hpp $(AUTOGRAD_DIR)/traced/
	cp traced/nn.hpp $(AUTOGRAD_DIR)/traced/
	cp traced/mlp.hpp $(AUTOGRAD_DIR)/traced/
	cp traced/mesh.hpp $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_mesh_traced.cpp $(AUTOGRAD_DIR)/main.cpp

bench-mesh-traced-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-mesh-traced-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Multi-Queue Benchmark: Test 4 command queues for parallel partition dispatch ===
bench-mesh-multiqueue: bench-mesh-multiqueue-sync bench-mesh-multiqueue-build bench-mesh-multiqueue-run

bench-mesh-multiqueue-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp traced/trace.hpp $(AUTOGRAD_DIR)/traced/
	cp traced/ops.hpp $(AUTOGRAD_DIR)/traced/
	cp traced/nn.hpp $(AUTOGRAD_DIR)/traced/
	cp traced/mlp.hpp $(AUTOGRAD_DIR)/traced/
	cp traced/mesh.hpp $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_mesh_multiqueue.cpp $(AUTOGRAD_DIR)/main.cpp

bench-mesh-multiqueue-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-mesh-multiqueue-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Shape Sweep Benchmark: Find optimal shapes for peak utilization ===
bench-shape-sweep: bench-shape-sweep-sync bench-shape-sweep-build bench-shape-sweep-run

bench-shape-sweep-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_shape_sweep.cpp $(AUTOGRAD_DIR)/main.cpp

bench-shape-sweep-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-shape-sweep-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Sub-Device Parallel Dispatch PoC ===
bench-subdevice-parallel: bench-subdevice-parallel-sync bench-subdevice-parallel-build bench-subdevice-parallel-run

bench-subdevice-parallel-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_subdevice_parallel.cpp $(AUTOGRAD_DIR)/main.cpp

bench-subdevice-parallel-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-subdevice-parallel-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Sub-Device Configuration Sweep (1-8 sub-devices) ===
bench-subdevice-sweep: bench-subdevice-sweep-sync bench-subdevice-sweep-build bench-subdevice-sweep-run

bench-subdevice-sweep-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_subdevice_sweep.cpp $(AUTOGRAD_DIR)/main.cpp

bench-subdevice-sweep-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-subdevice-sweep-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Scaling Sweep Benchmark: Batch x Dimension (for TTML comparison) ===
bench-sweep: bench-sweep-sync bench-sweep-build bench-sweep-run

bench-sweep-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_sweep.cpp $(AUTOGRAD_DIR)/main.cpp

bench-sweep-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-sweep-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Autograd Sweep: Compare all implementations across batch x dim ===
bench-autograd-sweep: bench-autograd-sweep-sync bench-autograd-sweep-build bench-autograd-sweep-run

bench-autograd-sweep-sync:
	mkdir -p $(AUTOGRAD_DIR)/dynamic
	mkdir -p $(AUTOGRAD_DIR)/static
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r dynamic/* $(AUTOGRAD_DIR)/dynamic/
	cp -r static/* $(AUTOGRAD_DIR)/static/
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_autograd_sweep.cpp $(AUTOGRAD_DIR)/main.cpp

bench-autograd-sweep-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-autograd-sweep-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Single Shape Benchmark: Test one batch×dim with fresh device ===
bench-single: bench-single-sync bench-single-build bench-single-run

bench-single-sync:
	mkdir -p $(AUTOGRAD_DIR)/dynamic
	mkdir -p $(AUTOGRAD_DIR)/static
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r dynamic/* $(AUTOGRAD_DIR)/dynamic/
	cp -r static/* $(AUTOGRAD_DIR)/static/
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_single_shape.cpp $(AUTOGRAD_DIR)/main.cpp

bench-single-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-single-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === TTNN 2-CQ Benchmark: Real ops with queue_id ===
bench-ttnn-2cq: bench-ttnn-2cq-sync bench-ttnn-2cq-build bench-ttnn-2cq-run

bench-ttnn-2cq-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_ttnn_2cq.cpp $(AUTOGRAD_DIR)/main.cpp

bench-ttnn-2cq-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-ttnn-2cq-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === CoreGrid Partition Sweep: Find diminishing returns for 2x28 vs 56-core ===
# Usage: BENCH_M=1024 BENCH_K=1024 BENCH_N=1024 make bench-partition-sweep-run
bench-partition-sweep: bench-partition-sweep-sync bench-partition-sweep-build bench-partition-sweep-run

bench-partition-sweep-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_partition_sweep.cpp $(AUTOGRAD_DIR)/main.cpp

bench-partition-sweep-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-partition-sweep-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Transformer Depth Benchmark: Trace region memory requirements ===
# Usage: BENCH_BATCH=32 BENCH_SEQ=128 BENCH_DIM=256 BENCH_LAYERS=8 make bench-transformer-depth-run
bench-transformer-depth: bench-transformer-depth-sync bench-transformer-depth-build bench-transformer-depth-run

bench-transformer-depth-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_transformer_depth.cpp $(AUTOGRAD_DIR)/main.cpp

bench-transformer-depth-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-transformer-depth-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === GPT-2 Small Benchmark: 12 stacked transformer layers ===
# Usage: BENCH_BATCH=32 BENCH_SEQ=512 make bench-gpt2-small
bench-gpt2-small: bench-gpt2-small-sync bench-gpt2-small-build bench-gpt2-small-run

bench-gpt2-small-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_gpt2_small.cpp $(AUTOGRAD_DIR)/main.cpp

bench-gpt2-small-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-gpt2-small-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

bench-gpt2-small-noc: bench-gpt2-small-sync bench-gpt2-small-build
	rm -rf ~/noc_traces_gpt2
	mkdir -p ~/noc_traces_gpt2
	cd $(TT_METAL) && \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
		TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=~/noc_traces_gpt2 \
		./build_Release/ttnn/examples/example_autograd
	mv ~/noc_traces_gpt2/cluster_coordinates.json ~/cluster_coordinates.json.bak 2>/dev/null || true
	@echo "Traces in ~/noc_traces_gpt2/"
	@echo "Sample analysis: ls ~/noc_traces_gpt2/*.json | head -20 | xargs -I{} cp {} ~/noc_traces_gpt2_sample/"

clean:
	rm -rf $(AUTOGRAD_DIR)
	rm -f $(BUILD_DIR)/ttnn/examples/example_autograd

# === Transformer Training Benchmark: full training step (forward + backward + SGD) ===
# Usage: BENCH_BATCH=32 BENCH_SEQ=256 BENCH_DIM=384 BENCH_HEADS=6 BENCH_LAYERS=6 make bench-transformer-train
.PHONY: bench-transformer-train bench-transformer-train-sync bench-transformer-train-build bench-transformer-train-run

bench-transformer-train: bench-transformer-train-sync bench-transformer-train-build bench-transformer-train-run

bench-transformer-train-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_transformer_train.cpp $(AUTOGRAD_DIR)/main.cpp

bench-transformer-train-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-transformer-train-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Full GPT-2 Benchmark (with embeddings) - Fair TTML comparison ===
# Usage: BENCH_LAYERS=6 make bench-gpt2-full
.PHONY: bench-gpt2-full bench-gpt2-full-sync bench-gpt2-full-build bench-gpt2-full-run

bench-gpt2-full: bench-gpt2-full-sync bench-gpt2-full-build bench-gpt2-full-run

bench-gpt2-full-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_gpt2_full.cpp $(AUTOGRAD_DIR)/main.cpp

bench-gpt2-full-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-gpt2-full-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === GPT-2 Static vs Traced Benchmark ===
# Compares static (non-traced) vs traced execution for full GPT-2 with embeddings
# Usage: BENCH_LAYERS=6 make bench-gpt2-trace
.PHONY: bench-gpt2-trace bench-gpt2-trace-sync bench-gpt2-trace-build bench-gpt2-trace-run

bench-gpt2-trace: bench-gpt2-trace-sync bench-gpt2-trace-build bench-gpt2-trace-run

bench-gpt2-trace-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_gpt2_trace.cpp $(AUTOGRAD_DIR)/main.cpp

bench-gpt2-trace-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-gpt2-trace-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd
