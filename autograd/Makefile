# C++ Autograd on TTNN
# Uses the tt-metal build system by copying files to example directories

TT_METAL := ~/tt-metal
BUILD_DIR := $(TT_METAL)/build_Release
TTNN_EXAMPLES := $(TT_METAL)/ttnn/examples
AUTOGRAD_DIR := $(TTNN_EXAMPLES)/autograd


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

# === Test: Typecast BFP8 <-> BF16 ===
typecast: typecast-sync typecast-build typecast-run

typecast-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp tests/test_typecast.cpp $(AUTOGRAD_DIR)/main.cpp

typecast-build:
	cd $(BUILD_DIR) && ninja example_autograd

typecast-run:
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
TT_METAL_DEVICE_PROFILER ?= 1
TT_METAL_DEVICE_PROFILER_SYNC ?= 1

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

# === GPT-2 Static Autograd Benchmark ===
# Tests PersistentGPT2 (static_autograd) with trace API
# This uses the correct pattern: Graph + Value* pointers to class member tensors
# Usage: BENCH_LAYERS=6 make bench-gpt2-static

bench-gpt2-static: bench-gpt2-static-sync bench-gpt2-static-build bench-gpt2-static-run

bench-gpt2-static-sync:
	mkdir -p $(AUTOGRAD_DIR)/static
	cp -r static/* $(AUTOGRAD_DIR)/static/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_gpt2_static.cpp $(AUTOGRAD_DIR)/main.cpp

bench-gpt2-static-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-gpt2-static-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# Profile BFP8 backward pass with Tracy

bench-gpt2-static-profile: bench-gpt2-static-sync bench-gpt2-static-build
	@echo "=== Tracy Profiling: GPT-2 BFP8 Backward Pass ==="
	@mkdir -p $(PROFILER_LOGS)
	@echo "Starting Tracy capture on port $(TRACY_PORT)..."
	@$(CAPTURE_BIN) -o $(PROFILER_LOGS)/gpt2_backward.tracy -f -p $(TRACY_PORT) & \
	CAPTURE_PID=$$!; \
	sleep 2; \
	echo "Running GPT-2 training with Tracy profiling..."; \
	cd $(TT_METAL) && TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_SYNC=1 TRACY_PORT=$(TRACY_PORT) \
		./build_Release/ttnn/examples/example_autograd; \
	sleep 1; \
	kill $$CAPTURE_PID 2>/dev/null || true; \
	wait $$CAPTURE_PID 2>/dev/null || true
	@echo "Exporting to CSV..."
	@$(CSVEXPORT_BIN) -u $(PROFILER_LOGS)/gpt2_backward.tracy \
		> $(PROFILER_LOGS)/gpt2_backward.csv 2>/dev/null || true
	@echo ""
	@echo "=== Profiling Complete ==="
	@echo "Tracy trace: $(PROFILER_LOGS)/gpt2_backward.tracy"
	@echo "CSV output:  $(PROFILER_LOGS)/gpt2_backward.csv"
	@echo ""
	@echo "Key zones to check in CSV:"
	@echo "  - graph_backward: Total backward pass time"
	@echo "  - backward_fn: Individual backward function calls"
	@echo "  - accumulate_grad: Gradient accumulation"
	@echo "  - accumulate_grad_add: Allocating add (should be minimized)"
	@echo "  - Linear3DBFP8_backward, LayerNormBFP8_backward: Layer-specific"
	@echo "  - typecast_dout_bf16, ln_typecast_dout: BFP8->BF16 conversions"

# === GPT-2 Forward+Backward Benchmark (BF16 only) ===
# Simple benchmark for single step timing
# Usage: BENCH_LAYERS=6 make bench-gpt2-fwd-back

bench-gpt2-fwd-back: bench-gpt2-fwd-back-sync bench-gpt2-fwd-back-build bench-gpt2-fwd-back-run

bench-gpt2-fwd-back-sync:
	mkdir -p $(AUTOGRAD_DIR)/static
	cp -r static/* $(AUTOGRAD_DIR)/static/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_gpt2_fwd_back.cpp $(AUTOGRAD_DIR)/main.cpp

bench-gpt2-fwd-back-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-gpt2-fwd-back-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === GPT-2 Mixed Precision Benchmark (BF16 Attn + BFP8 FFN) ===
# Compares pure BF16 vs mixed precision training
# Usage: BENCH_LAYERS=6 make bench-gpt2-fwd-back-mxp

bench-gpt2-fwd-back-mxp: bench-gpt2-fwd-back-mxp-sync bench-gpt2-fwd-back-mxp-build bench-gpt2-fwd-back-mxp-run

bench-gpt2-fwd-back-mxp-sync:
	mkdir -p $(AUTOGRAD_DIR)/static
	cp -r static/* $(AUTOGRAD_DIR)/static/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../bench/autograd/bench_gpt2_fwd_back_mxp.cpp $(AUTOGRAD_DIR)/main.cpp

bench-gpt2-fwd-back-mxp-build:
	cd $(BUILD_DIR) && ninja example_autograd

bench-gpt2-fwd-back-mxp-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# Profile mixed precision with Tracy

bench-gpt2-fwd-back-mxp-profile: bench-gpt2-fwd-back-mxp-sync bench-gpt2-fwd-back-mxp-build
	@echo "=== Tracy Profiling: GPT-2 Mixed Precision ==="
	@mkdir -p $(PROFILER_LOGS)
	@echo "Starting Tracy capture on port $(TRACY_PORT)..."
	@$(CAPTURE_BIN) -o $(PROFILER_LOGS)/gpt2_mxp.tracy -f -p $(TRACY_PORT) & \
	CAPTURE_PID=$$!; \
	sleep 2; \
	echo "Running mixed precision benchmark with Tracy profiling..."; \
	cd $(TT_METAL) && TT_METAL_DEVICE_PROFILER=1 TT_METAL_DEVICE_PROFILER_SYNC=1 TRACY_PORT=$(TRACY_PORT) \
		./build_Release/ttnn/examples/example_autograd; \
	sleep 1; \
	kill $$CAPTURE_PID 2>/dev/null || true; \
	wait $$CAPTURE_PID 2>/dev/null || true
	@echo "Exporting to CSV..."
	@$(CSVEXPORT_BIN) -u $(PROFILER_LOGS)/gpt2_mxp.tracy \
		> $(PROFILER_LOGS)/gpt2_mxp.csv 2>/dev/null || true
	@echo ""
	@echo "=== Profiling Complete ==="
	@echo "Tracy trace: $(PROFILER_LOGS)/gpt2_mxp.tracy"
	@echo "CSV output:  $(PROFILER_LOGS)/gpt2_mxp.csv"

# === PyTorch Benchmarks on adam (CUDA) ===
# Run via SSH from howard to adam
ADAM := adam
ADAM_DIR := ~/ttnn-perf
CUDA_RESULTS := ../bench/reports/cuda

# === tensordiff: TTNN↔Torch Tensor Comparison ===
# Usage: make tensordiff-compare

tensordiff-compare: tensordiff-compare-sync tensordiff-compare-build tensordiff-compare-run

tensordiff-compare-sync:
	mkdir -p $(AUTOGRAD_DIR)/tensordiff
	cp -r ../tensordiff/core $(AUTOGRAD_DIR)/tensordiff/
	cp ../tensordiff/compare_ttnn_torch.cpp $(AUTOGRAD_DIR)/main.cpp

tensordiff-compare-build:
	cd $(BUILD_DIR) && ninja example_autograd

tensordiff-compare-run:
	cd $(TT_METAL) && TENSORDIFF_OUT_DIR=$(HOME)/ttnn-perf/tensordiff/outputs/ttnn \
		./build_Release/ttnn/examples/example_autograd

# === tensordiff: BF16 matmul diff (TTNN) ===
# Usage: make tensordiff-matmul

tensordiff-matmul: tensordiff-matmul-sync tensordiff-matmul-build tensordiff-matmul-run

tensordiff-matmul-sync:
	mkdir -p $(AUTOGRAD_DIR)/tensordiff
	cp -r ../tensordiff/core $(AUTOGRAD_DIR)/tensordiff/
	cp ../tensordiff/matmul_bf16_diff.cpp $(AUTOGRAD_DIR)/main.cpp

tensordiff-matmul-build:
	cd $(BUILD_DIR) && ninja example_autograd

tensordiff-matmul-run:
	cd $(TT_METAL) && TENSORDIFF_OUT_DIR=$(HOME)/ttnn-perf/tensordiff/outputs/matmul \
		./build_Release/ttnn/examples/example_autograd

# === tensordiff: Torch random init + TTNN matmul from TDF1 ===
# Usage: make tensordiff-matmul-torch

tensordiff-matmul-torch: tensordiff-matmul-torch-gen tensordiff-matmul-torch-ttnn

tensordiff-matmul-torch-gen: tensordiff-torch-compare-build
	cd $(TENSORDIFF_BUILD_DIR) && TENSORDIFF_OUT_DIR=$(HOME)/ttnn-perf/tensordiff/outputs/matmul_torch_rand \
		./tensordiff_torch_random_init

tensordiff-matmul-torch-ttnn: tensordiff-matmul-torch-ttnn-sync tensordiff-matmul-torch-ttnn-build tensordiff-matmul-torch-ttnn-run

tensordiff-matmul-torch-ttnn-sync:
	mkdir -p $(AUTOGRAD_DIR)/tensordiff
	cp -r ../tensordiff/core $(AUTOGRAD_DIR)/tensordiff/
	cp ../tensordiff/matmul_from_tdf1.cpp $(AUTOGRAD_DIR)/main.cpp

tensordiff-matmul-torch-ttnn-build:
	cd $(BUILD_DIR) && ninja example_autograd

tensordiff-matmul-torch-ttnn-run:
	cd $(TT_METAL) && TENSORDIFF_OUT_DIR=$(HOME)/ttnn-perf/tensordiff/outputs/matmul_torch_rand \
		./build_Release/ttnn/examples/example_autograd

# === tensordiff: LayerNorm (Torch → TTNN) ===
# Usage: make tensordiff-layernorm

tensordiff-layernorm: tensordiff-layernorm-gen tensordiff-layernorm-ttnn

tensordiff-layernorm-gen: tensordiff-torch-compare-build
	cd $(TENSORDIFF_BUILD_DIR) && TENSORDIFF_OUT_DIR=$(HOME)/ttnn-perf/tensordiff/outputs/layernorm_torch_rand \
		./tensordiff_torch_layernorm_init

tensordiff-layernorm-ttnn: tensordiff-layernorm-ttnn-sync tensordiff-layernorm-ttnn-build tensordiff-layernorm-ttnn-run

tensordiff-layernorm-ttnn-sync:
	mkdir -p $(AUTOGRAD_DIR)/tensordiff
	cp -r ../tensordiff/core $(AUTOGRAD_DIR)/tensordiff/
	cp ../tensordiff/layernorm_from_tdf1.cpp $(AUTOGRAD_DIR)/main.cpp

tensordiff-layernorm-ttnn-build:
	cd $(BUILD_DIR) && ninja example_autograd

tensordiff-layernorm-ttnn-run:
	cd $(TT_METAL) && TENSORDIFF_OUT_DIR=$(HOME)/ttnn-perf/tensordiff/outputs/layernorm_torch_rand \
		./build_Release/ttnn/examples/example_autograd

# === tensordiff: Transformer layer (Torch → TTNN) ===
# Usage: make tensordiff-transformer-layer

tensordiff-transformer-layer: tensordiff-transformer-layer-gen tensordiff-transformer-layer-ttnn

tensordiff-transformer-layer-gen: tensordiff-torch-compare-build
	cd $(TENSORDIFF_BUILD_DIR) && TENSORDIFF_OUT_DIR=$(HOME)/ttnn-perf/tensordiff/outputs/transformer_layer_torch_rand \
		./tensordiff_torch_transformer_layer_init

tensordiff-transformer-layer-ttnn: tensordiff-transformer-layer-ttnn-sync tensordiff-transformer-layer-ttnn-build tensordiff-transformer-layer-ttnn-run

tensordiff-transformer-layer-ttnn-sync:
	mkdir -p $(AUTOGRAD_DIR)/tensordiff
	cp -r ../tensordiff/core $(AUTOGRAD_DIR)/tensordiff/
	cp ../tensordiff/transformer_layer_from_tdf1.cpp $(AUTOGRAD_DIR)/main.cpp

tensordiff-transformer-layer-ttnn-build:
	cd $(BUILD_DIR) && ninja example_autograd

tensordiff-transformer-layer-ttnn-run:
	cd $(TT_METAL) && TENSORDIFF_OUT_DIR=$(HOME)/ttnn-perf/tensordiff/outputs/transformer_layer_torch_rand \
		./build_Release/ttnn/examples/example_autograd

# === tensordiff: Transformer layer backward (Torch → TTNN autograd) ===
# Usage: make tensordiff-transformer-layer-bwd

tensordiff-transformer-layer-bwd: tensordiff-transformer-layer-bwd-gen tensordiff-transformer-layer-bwd-ttnn

tensordiff-transformer-layer-bwd-gen: tensordiff-torch-compare-build
	cd $(TENSORDIFF_BUILD_DIR) && TENSORDIFF_OUT_DIR=$(HOME)/ttnn-perf/tensordiff/outputs/transformer_layer_bwd_torch_rand \
		./tensordiff_torch_transformer_layer_backward_init

tensordiff-transformer-layer-bwd-ttnn: tensordiff-transformer-layer-bwd-ttnn-sync tensordiff-transformer-layer-bwd-ttnn-build tensordiff-transformer-layer-bwd-ttnn-run

tensordiff-transformer-layer-bwd-ttnn-sync:
	mkdir -p $(AUTOGRAD_DIR)/tensordiff
	mkdir -p $(AUTOGRAD_DIR)/autograd
	cp -r ../tensordiff/core $(AUTOGRAD_DIR)/tensordiff/
	cp -r ../autograd/nn $(AUTOGRAD_DIR)/autograd/
	cp ../autograd/ops.hpp ../autograd/value.hpp ../autograd/tensor_group.hpp $(AUTOGRAD_DIR)/autograd/
	cp ../tensordiff/transformer_layer_backward_from_tdf1.cpp $(AUTOGRAD_DIR)/main.cpp

tensordiff-transformer-layer-bwd-ttnn-build:
	cd $(BUILD_DIR) && ninja example_autograd

tensordiff-transformer-layer-bwd-ttnn-run:
	cd $(TT_METAL) && TENSORDIFF_OUT_DIR=$(HOME)/ttnn-perf/tensordiff/outputs/transformer_layer_bwd_torch_rand \
		./build_Release/ttnn/examples/example_autograd

# === tensordiff: Softmax backward (binary op) ===
# Usage: make tensordiff-softmax-bw

tensordiff-softmax-bw: tensordiff-transformer-layer-bwd-gen tensordiff-softmax-bw-ttnn

tensordiff-softmax-bw-ttnn: tensordiff-softmax-bw-ttnn-sync tensordiff-softmax-bw-ttnn-build tensordiff-softmax-bw-ttnn-run

tensordiff-softmax-bw-ttnn-sync:
	mkdir -p $(AUTOGRAD_DIR)/tensordiff
	cp -r ../tensordiff/core $(AUTOGRAD_DIR)/tensordiff/
	cp ../tensordiff/softmax_backward_from_tdf1.cpp $(AUTOGRAD_DIR)/main.cpp

tensordiff-softmax-bw-ttnn-build:
	cd $(BUILD_DIR) && ninja example_autograd

tensordiff-softmax-bw-ttnn-run:
	cd $(TT_METAL) && TENSORDIFF_OUT_DIR=$(HOME)/ttnn-perf/tensordiff/outputs/transformer_layer_bwd_torch_rand \
		./build_Release/ttnn/examples/example_autograd

# === tensordiff: libtorch comparator (CPU) ===
# Requires libtorch installed at ~/libtorch
# Usage: make tensordiff-torch-compare

TENSORDIFF_DIR := ../tensordiff
TENSORDIFF_BUILD_DIR := $(TENSORDIFF_DIR)/build
TENSORDIFFGRAD_DIR := ../tensordiffgrad
TENSORDIFFGRAD_BUILD_DIR := $(TENSORDIFFGRAD_DIR)/build
RECORDER_TEST1_DIR := ../experiments/recorder_test1
RECORDER_TEST1_BUILD_DIR := $(RECORDER_TEST1_DIR)/build

tensordiff-torch-compare: tensordiff-torch-compare-build tensordiff-torch-compare-run

tensordiff-torch-compare-build:
	mkdir -p $(TENSORDIFF_BUILD_DIR)
	cd $(TENSORDIFF_BUILD_DIR) && \
		TT_METAL_HOME=$(TT_METAL) cmake -DCMAKE_PREFIX_PATH=$(HOME)/libtorch .. && \
		$(MAKE)

tensordiff-torch-compare-run:
	cd $(TENSORDIFF_BUILD_DIR) && \
		TT_METAL_HOME=$(TT_METAL) LD_LIBRARY_PATH=$(HOME)/libtorch/lib:$$LD_LIBRARY_PATH \
		./tensordiff_torch_compare --dir $(HOME)/ttnn-perf/tensordiff/outputs/ttnn

# === tensordiff: tensor ops test (TTNN) ===
# Usage: make tensordiff-ops-test

tensordiff-ops-test: tensordiff-ops-test-sync tensordiff-ops-test-build tensordiff-ops-test-run

tensordiff-ops-test-sync:
	mkdir -p $(AUTOGRAD_DIR)/tensordiff
	cp -r ../tensordiff/core $(AUTOGRAD_DIR)/tensordiff/
	cp ../tensordiff/ttnn_tensor.hpp $(AUTOGRAD_DIR)/tensordiff/
	cp ../tensordiff/tests/tensor_ops_test.cpp $(AUTOGRAD_DIR)/main.cpp

tensordiff-ops-test-build:
	cd $(BUILD_DIR) && ninja example_autograd

tensordiff-ops-test-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === tensordiff: BaseTensor dispatch test (TTNN) ===
# Usage: make tensordiff-dispatch-test

tensordiff-dispatch-test: tensordiff-dispatch-test-sync tensordiff-dispatch-test-build tensordiff-dispatch-test-run

tensordiff-dispatch-test-sync:
	mkdir -p $(AUTOGRAD_DIR)/tensordiff
	cp -r ../tensordiff/core $(AUTOGRAD_DIR)/tensordiff/
	cp ../tensordiff/ttnn_tensor.hpp $(AUTOGRAD_DIR)/tensordiff/
	cp ../tensordiff/tests/base_tensor_dispatch_test.cpp $(AUTOGRAD_DIR)/main.cpp

tensordiff-dispatch-test-build:
	cd $(BUILD_DIR) && ninja example_autograd

tensordiff-dispatch-test-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === tensordiff: tensor ops test (Torch/libtorch) ===
# Usage: make tensordiff-ops-test-torch

tensordiff-ops-test-torch: tensordiff-ops-test-torch-build tensordiff-ops-test-torch-run

tensordiff-ops-test-torch-build:
	mkdir -p $(TENSORDIFF_BUILD_DIR)
	cd $(TENSORDIFF_BUILD_DIR) && \
		TT_METAL_HOME=$(TT_METAL) cmake -DCMAKE_PREFIX_PATH=$(HOME)/libtorch .. && \
		$(MAKE) tensordiff_tensor_ops_test

tensordiff-ops-test-torch-run:
	cd $(TENSORDIFF_BUILD_DIR) && \
		TT_METAL_HOME=$(TT_METAL) LD_LIBRARY_PATH=$(HOME)/libtorch/lib:$$LD_LIBRARY_PATH \
		./tensordiff_tensor_ops_test

# === tensordiffgrad: tensor ops test (TTNN) ===
# Usage: make tensordiffgrad-ops-test

tensordiffgrad-ops-test: tensordiffgrad-ops-test-sync tensordiffgrad-ops-test-build tensordiffgrad-ops-test-run

tensordiffgrad-ops-test-sync:
	mkdir -p $(AUTOGRAD_DIR)/tensordiff
	cp -r ../tensordiff/core $(AUTOGRAD_DIR)/tensordiff/
	cp ../tensordiff/ttnn_tensor.hpp $(AUTOGRAD_DIR)/tensordiff/
	cp ../tensordiffgrad/tests/backend_ops_test.cpp $(AUTOGRAD_DIR)/main.cpp

tensordiffgrad-ops-test-build:
	cd $(BUILD_DIR) && ninja example_autograd

tensordiffgrad-ops-test-run:
	cd $(TT_METAL) && TENSORDIFF_OPLOG_DIR=$(HOME)/ttnn-perf/tensordiffgrad/outputs/oplog \
		./build_Release/ttnn/examples/example_autograd

# === tensordiffgrad: tensor ops test (Torch/libtorch) ===
# Usage: make tensordiffgrad-ops-test-torch

tensordiffgrad-ops-test-torch: tensordiffgrad-ops-test-torch-build tensordiffgrad-ops-test-torch-run

tensordiffgrad-ops-test-torch-build:
	mkdir -p $(TENSORDIFFGRAD_BUILD_DIR)
	cd $(TENSORDIFFGRAD_BUILD_DIR) && \
		TT_METAL_HOME=$(TT_METAL) cmake -DCMAKE_PREFIX_PATH=$(HOME)/libtorch .. && \
		$(MAKE) tensordiffgrad_torch_ops_test

tensordiffgrad-ops-test-torch-run:
	cd $(TENSORDIFFGRAD_BUILD_DIR) && \
		TT_METAL_HOME=$(TT_METAL) TENSORDIFF_OPLOG_DIR=$(HOME)/ttnn-perf/tensordiffgrad/outputs/oplog_torch \
		LD_LIBRARY_PATH=$(HOME)/libtorch/lib:$$LD_LIBRARY_PATH \
		./tensordiffgrad_torch_ops_test

# === tensordiffgrad: FFN test (Torch/libtorch, CPU) ===
# Usage: make tensordiffgrad-ffn-test-torch

tensordiffgrad-ffn-test-torch: tensordiffgrad-ffn-test-torch-build tensordiffgrad-ffn-test-torch-run

tensordiffgrad-ffn-test-torch-build:
	mkdir -p $(TENSORDIFFGRAD_BUILD_DIR)
	cd $(TENSORDIFFGRAD_BUILD_DIR) && \
		TT_METAL_HOME=$(TT_METAL) cmake -DCMAKE_PREFIX_PATH=$(HOME)/libtorch .. && \
		$(MAKE) tensordiffgrad_torch_ffn_test

tensordiffgrad-ffn-test-torch-run:
	cd $(TENSORDIFFGRAD_BUILD_DIR) && \
		TT_METAL_HOME=$(TT_METAL) TENSORDIFF_OPLOG_DIR=$(HOME)/ttnn-perf/experiments/ffn_forward/oplog_torch \
		LD_LIBRARY_PATH=$(HOME)/libtorch/lib:$$LD_LIBRARY_PATH \
		TENSORDIFFGRAD_OUT_DIR=$(HOME)/ttnn-perf/experiments/ffn_forward/torch \
		./tensordiffgrad_torch_ffn_test

# === TTNN FFN forward test (static_autograd ops) ===
# Usage: make ffn-forward

ffn-forward: ffn-forward-sync ffn-forward-build ffn-forward-run

ffn-forward-sync:
	mkdir -p $(AUTOGRAD_DIR)/tensordiff
	cp -r ../tensordiff/core $(AUTOGRAD_DIR)/tensordiff/
	cp ../autograd/value.hpp $(AUTOGRAD_DIR)/
	cp ../autograd/ops.hpp $(AUTOGRAD_DIR)/
	cp ../autograd/tests/test_ffn_forward.cpp $(AUTOGRAD_DIR)/main.cpp

ffn-forward-build:
	cd $(BUILD_DIR) && ninja example_autograd

ffn-forward-run:
	cd $(TT_METAL) && TENSORDIFF_OPLOG_DIR=$(HOME)/ttnn-perf/experiments/ffn_forward/oplog \
		./build_Release/ttnn/examples/example_autograd

# === tensordiff: BaseTensor dispatch test (Torch/libtorch) ===
# Usage: make tensordiff-dispatch-test-torch

tensordiff-dispatch-test-torch: tensordiff-dispatch-test-torch-build tensordiff-dispatch-test-torch-run

tensordiff-dispatch-test-torch-build:
	mkdir -p $(TENSORDIFF_BUILD_DIR)
	cd $(TENSORDIFF_BUILD_DIR) && \
		TT_METAL_HOME=$(TT_METAL) cmake -DCMAKE_PREFIX_PATH=$(HOME)/libtorch .. && \
		$(MAKE) tensordiff_tensor_dispatch_test

tensordiff-dispatch-test-torch-run:
	cd $(TENSORDIFF_BUILD_DIR) && \
		TT_METAL_HOME=$(TT_METAL) LD_LIBRARY_PATH=$(HOME)/libtorch/lib:$$LD_LIBRARY_PATH \
		./tensordiff_tensor_dispatch_test


# Run GPT-2 PyTorch on adam (CUDA), save results to howard
bench-gpt2-torch: bench-gpt2-torch-run

bench-gpt2-torch-run:
	@mkdir -p $(CUDA_RESULTS)
	ssh $(ADAM) "cd $(ADAM_DIR) && source .venv/bin/activate && python3 bench/gpt2_torch.py" 2>&1 | tee $(CUDA_RESULTS)/gpt2_torch.txt
	@echo ""
	@echo "Results saved to $(CUDA_RESULTS)/gpt2_torch.txt"

# === Forward Pass Verification (C++ Traced Autograd vs PyTorch) ===
# Compares our C++ traced autograd outputs against PyTorch BF16 reference
# Uses file-based serialization for exact numerical comparison
#
# Usage:
#   make verify-attention          # Full pipeline: generate + run + compare
#   make verify-attention-pytorch  # Generate PyTorch reference only
#   make verify-attention-cpp      # Build and run C++ only
#   make verify-attention-compare  # Compare outputs only
#
VERIFY_DIR := ../verify


# Full verification pipeline
verify-attention: verify-attention-pytorch verify-attention-cpp verify-attention-compare

# Generate PyTorch reference outputs
verify-attention-pytorch:
	@echo "=== Generating PyTorch reference outputs ==="
	cd .. && python3 verify/gen_pytorch_outputs.py

# Build and run C++ verification
verify-attention-cpp: verify-attention-sync verify-attention-build verify-attention-run

verify-attention-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp $(VERIFY_DIR)/verify_attention.cpp $(AUTOGRAD_DIR)/main.cpp

verify-attention-build:
	cd $(BUILD_DIR) && ninja example_autograd

verify-attention-run:
	@echo "=== Running C++ traced autograd forward ==="
	cd ~/ttnn-perf && TT_METAL_HOME=~/tt-metal TT_METAL_RUNTIME_ROOT=~/tt-metal ~/tt-metal/build_Release/ttnn/examples/example_autograd

# Compare outputs
verify-attention-compare:
	@echo "=== Comparing C++ vs PyTorch outputs ==="
	cd .. && python3 verify/compare_outputs.py

# === FP32 Accumulation Ablation ===
# Compares attention forward with fp32_dest_acc_en=true vs PyTorch
#
# Usage:
#   make verify-attention-fp32acc          # Full pipeline
#   make verify-attention-fp32acc-cpp      # Build and run C++ only
#   make verify-attention-fp32acc-compare  # Compare outputs only
#

# Full verification pipeline with FP32 accumulation
verify-attention-fp32acc: verify-attention-pytorch verify-attention-fp32acc-cpp verify-attention-fp32acc-compare

# Build and run C++ verification with FP32 accumulation
verify-attention-fp32acc-cpp: verify-attention-fp32acc-sync verify-attention-build verify-attention-fp32acc-run

verify-attention-fp32acc-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp $(VERIFY_DIR)/verify_attention_fp32acc.cpp $(AUTOGRAD_DIR)/main.cpp

verify-attention-fp32acc-run:
	@echo "=== Running C++ traced autograd forward (FP32 ACC) ==="
	cd ~/ttnn-perf && TT_METAL_HOME=~/tt-metal TT_METAL_RUNTIME_ROOT=~/tt-metal ~/tt-metal/build_Release/ttnn/examples/example_autograd

# Compare FP32 accumulation outputs
verify-attention-fp32acc-compare:
	@echo "=== Comparing C++ (FP32 acc) vs PyTorch outputs ==="
	cd .. && python3 verify/compare_outputs.py --cpp-dir verify/outputs/cpp_fp32acc

# === Per-Layer Timing ===
# Measures actual TFLOPS for each attention layer
#
# Usage:
#   make verify-attention-timed
#

verify-attention-timed: verify-attention-timed-sync verify-attention-build verify-attention-timed-run

verify-attention-timed-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp $(VERIFY_DIR)/verify_attention_timed.cpp $(AUTOGRAD_DIR)/main.cpp

verify-attention-timed-run:
	@echo "=== Running C++ attention timing ==="
	cd ~/ttnn-perf && TT_METAL_HOME=~/tt-metal TT_METAL_RUNTIME_ROOT=~/tt-metal ~/tt-metal/build_Release/ttnn/examples/example_autograd

# === SDPA vs Manual Attention Comparison ===
# Compare fused SDPA kernel against our manual implementation
# Usage:
#   make verify-attention-sdpa
#

verify-attention-sdpa: verify-attention-pytorch verify-attention-sdpa-sync verify-attention-build verify-attention-sdpa-run

verify-attention-sdpa-sync:
	mkdir -p $(AUTOGRAD_DIR)/traced
	cp -r traced/* $(AUTOGRAD_DIR)/traced/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp $(VERIFY_DIR)/verify_attention_sdpa.cpp $(AUTOGRAD_DIR)/main.cpp

verify-attention-sdpa-run:
	@echo "=== Running SDPA vs Manual comparison ==="
	cd ~/ttnn-perf && TT_METAL_HOME=~/tt-metal TT_METAL_RUNTIME_ROOT=~/tt-metal ~/tt-metal/build_Release/ttnn/examples/example_autograd

# === Grokking: Modular Division Experiment ===
# Usage: GROK_CONFIG=~/ttnn-perf/experiments/grok/default.yaml make grok-train

grok-train: grok-train-sync grok-train-build grok-train-run

grok-train-sync:
	mkdir -p $(AUTOGRAD_DIR)/autograd
	mkdir -p $(AUTOGRAD_DIR)/data
	mkdir -p $(AUTOGRAD_DIR)/utils
	cp -r nn $(AUTOGRAD_DIR)/autograd/
	cp -r utils $(AUTOGRAD_DIR)/autograd/
	cp ops.hpp tensor_group.hpp value.hpp nn.hpp $(AUTOGRAD_DIR)/autograd/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../data/grok.hpp ../data/grok.cpp $(AUTOGRAD_DIR)/data/
	cp ../experiments/grok/train_grok.cpp $(AUTOGRAD_DIR)/main.cpp

grok-train-build:
	cd $(BUILD_DIR) && ninja example_autograd

grok-train-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Linear Verify (BF16) ===
# Usage: LINEAR_VERIFY_CONFIG=~/ttnn-perf/experiments/linear_verify/default.yaml make linear-verify

linear-verify: linear-verify-sync linear-verify-build linear-verify-run

linear-verify-sync:
	mkdir -p $(AUTOGRAD_DIR)/autograd
	mkdir -p $(AUTOGRAD_DIR)/utils
	cp -r nn $(AUTOGRAD_DIR)/autograd/
	cp -r utils $(AUTOGRAD_DIR)/autograd/
	cp ops.hpp tensor_group.hpp value.hpp nn.hpp $(AUTOGRAD_DIR)/autograd/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../experiments/linear_verify/linear_verify.cpp $(AUTOGRAD_DIR)/main.cpp

linear-verify-build:
	cd $(BUILD_DIR) && ninja example_autograd

linear-verify-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Softmax Backward Verify (BF16) ===
# Usage: SOFTMAX_BW_CONFIG=~/ttnn-perf/experiments/grok/softmax_bw_verify.yaml make softmax-bw-verify

softmax-bw-verify: softmax-bw-verify-sync softmax-bw-verify-build softmax-bw-verify-run

softmax-bw-verify-sync:
	mkdir -p $(AUTOGRAD_DIR)/autograd
	mkdir -p $(AUTOGRAD_DIR)/utils
	cp -r nn $(AUTOGRAD_DIR)/autograd/
	cp -r utils $(AUTOGRAD_DIR)/autograd/
	cp ops.hpp tensor_group.hpp value.hpp nn.hpp $(AUTOGRAD_DIR)/autograd/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../experiments/grok/softmax_bw_verify.cpp $(AUTOGRAD_DIR)/main.cpp

softmax-bw-verify-build:
	cd $(BUILD_DIR) && ninja example_autograd

softmax-bw-verify-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Embedding Backward Verify (BF16) ===
# Usage: EMBEDDING_BW_CONFIG=~/ttnn-perf/experiments/grok/embedding_bw_verify.yaml make embedding-bw-verify

embedding-bw-verify: embedding-bw-verify-sync embedding-bw-verify-build embedding-bw-verify-run

embedding-bw-verify-sync:
	mkdir -p $(AUTOGRAD_DIR)/autograd
	mkdir -p $(AUTOGRAD_DIR)/utils
	cp -r nn $(AUTOGRAD_DIR)/autograd/
	cp -r utils $(AUTOGRAD_DIR)/autograd/
	cp ops.hpp tensor_group.hpp value.hpp nn.hpp $(AUTOGRAD_DIR)/autograd/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../experiments/grok/embedding_bw_verify.cpp $(AUTOGRAD_DIR)/main.cpp

embedding-bw-verify-build:
	cd $(BUILD_DIR) && ninja example_autograd

embedding-bw-verify-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Attention Backward Verify (BF16) ===
# Usage: ATTN_BW_CONFIG=~/ttnn-perf/experiments/grok/attn_bw_verify.yaml make attn-bw-verify

attn-bw-verify: attn-bw-verify-sync attn-bw-verify-build attn-bw-verify-run

attn-bw-verify-sync:
	mkdir -p $(AUTOGRAD_DIR)/autograd
	mkdir -p $(AUTOGRAD_DIR)/utils
	cp -r nn $(AUTOGRAD_DIR)/autograd/
	cp -r utils $(AUTOGRAD_DIR)/autograd/
	cp ops.hpp tensor_group.hpp value.hpp nn.hpp $(AUTOGRAD_DIR)/autograd/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../experiments/grok/attn_bw_verify.cpp $(AUTOGRAD_DIR)/main.cpp

attn-bw-verify-build:
	cd $(BUILD_DIR) && ninja example_autograd

attn-bw-verify-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === GELU Backward Verify (BF16) ===
# Usage: GELU_BW_CONFIG=~/ttnn-perf/experiments/grok/gelu_bw_verify.yaml make gelu-bw-verify

gelu-bw-verify: gelu-bw-verify-sync gelu-bw-verify-build gelu-bw-verify-run

gelu-bw-verify-sync:
	mkdir -p $(AUTOGRAD_DIR)/autograd
	mkdir -p $(AUTOGRAD_DIR)/utils
	cp -r nn $(AUTOGRAD_DIR)/autograd/
	cp -r utils $(AUTOGRAD_DIR)/autograd/
	cp ops.hpp tensor_group.hpp value.hpp nn.hpp $(AUTOGRAD_DIR)/autograd/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../experiments/grok/gelu_bw_verify.cpp $(AUTOGRAD_DIR)/main.cpp

gelu-bw-verify-build:
	cd $(BUILD_DIR) && ninja example_autograd

gelu-bw-verify-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === MNIST MLP Parity (AdamW) ===
# Usage: MNIST_CONFIG=~/ttnn-perf/experiments/mnist/default.yaml make mnist-train

mnist-train: mnist-train-sync mnist-train-build mnist-train-run

mnist-train-sync:
	mkdir -p $(AUTOGRAD_DIR)/autograd
	mkdir -p $(AUTOGRAD_DIR)/utils
	cp -r nn $(AUTOGRAD_DIR)/autograd/
	cp -r utils $(AUTOGRAD_DIR)/autograd/
	cp ops.hpp tensor_group.hpp value.hpp nn.hpp $(AUTOGRAD_DIR)/autograd/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../experiments/mnist/train_mnist.cpp $(AUTOGRAD_DIR)/main.cpp

mnist-train-build:
	cd $(BUILD_DIR) && ninja example_autograd

mnist-train-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === GPT2 Reimplementation Parity (single layer, op-by-op) ===
# Usage: GPT2_REIMPL_CONFIG=~/ttnn-perf/experiments/gpt2_reimpl/default.yaml make gpt2-reimpl

gpt2-reimpl: gpt2-reimpl-sync gpt2-reimpl-build gpt2-reimpl-run

gpt2-reimpl-sync:
	mkdir -p $(AUTOGRAD_DIR)/autograd
	mkdir -p $(AUTOGRAD_DIR)/utils
	cp -r nn $(AUTOGRAD_DIR)/autograd/
	cp -r utils $(AUTOGRAD_DIR)/autograd/
	cp ops.hpp tensor_group.hpp value.hpp nn.hpp $(AUTOGRAD_DIR)/autograd/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../experiments/gpt2_reimpl/gpt2_reimpl.cpp $(AUTOGRAD_DIR)/main.cpp

gpt2-reimpl-build:
	cd $(BUILD_DIR) && ninja example_autograd

gpt2-reimpl-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === GPT2 Rebuild Parity (incremental stages) ===
# Usage: GPT2_REBUILD_CONFIG=~/ttnn-perf/experiments/gpt2_rebuild/default.yaml make gpt2-rebuild

gpt2-rebuild: gpt2-rebuild-sync gpt2-rebuild-build gpt2-rebuild-run

gpt2-rebuild-sync:
	mkdir -p $(AUTOGRAD_DIR)/autograd
	mkdir -p $(AUTOGRAD_DIR)/utils
	cp -r nn $(AUTOGRAD_DIR)/autograd/
	cp -r utils $(AUTOGRAD_DIR)/autograd/
	cp ops.hpp tensor_group.hpp value.hpp nn.hpp $(AUTOGRAD_DIR)/autograd/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../experiments/gpt2_rebuild/gpt2_rebuild.cpp $(AUTOGRAD_DIR)/main.cpp

gpt2-rebuild-build:
	cd $(BUILD_DIR) && ninja example_autograd

gpt2-rebuild-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Recorder Test 1 (TTNN) ===
# Usage: RECORDER_TEST1_CONFIG=~/ttnn-perf/experiments/recorder_test1/recorder_test1.yaml make recorder-test1-ttnn

recorder-test1-ttnn: recorder-test1-ttnn-sync recorder-test1-ttnn-build recorder-test1-ttnn-run

recorder-test1-ttnn-sync:
	mkdir -p $(AUTOGRAD_DIR)/autograd
	mkdir -p $(AUTOGRAD_DIR)/utils
	mkdir -p $(AUTOGRAD_DIR)/tensordiff
	cp -r nn $(AUTOGRAD_DIR)/autograd/
	cp -r utils $(AUTOGRAD_DIR)/autograd/
	cp ops.hpp tensor_group.hpp value.hpp nn.hpp $(AUTOGRAD_DIR)/autograd/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp -r ../tensordiff/core $(AUTOGRAD_DIR)/tensordiff/
	cp ../experiments/recorder_test1/recorder_test1_ttnn.cpp $(AUTOGRAD_DIR)/main.cpp

recorder-test1-ttnn-build:
	cd $(BUILD_DIR) && ninja example_autograd

recorder-test1-ttnn-run:
	cd $(TT_METAL) && RECORDER_TEST1_CONFIG=$(HOME)/ttnn-perf/experiments/recorder_test1/recorder_test1.yaml \
		./build_Release/ttnn/examples/example_autograd

# === Recorder Test 1 (Torch C++) ===
# Usage: make recorder-test1-torch

recorder-test1-torch: recorder-test1-torch-build recorder-test1-torch-run

recorder-test1-torch-build:
	mkdir -p $(RECORDER_TEST1_BUILD_DIR)
	cd $(RECORDER_TEST1_BUILD_DIR) && \
		TT_METAL_HOME=$(TT_METAL) cmake -DCMAKE_PREFIX_PATH=$(HOME)/libtorch .. && \
		$(MAKE) recorder_test1_torch

recorder-test1-torch-run:
	cd $(RECORDER_TEST1_BUILD_DIR) && \
		TT_METAL_HOME=$(TT_METAL) LD_LIBRARY_PATH=$(HOME)/libtorch/lib:$$LD_LIBRARY_PATH \
		./recorder_test1_torch

# === Train Folder: GPT2 Rebuild (incremental stages) ===
# Usage: TRAIN_GPT2_CONFIG=~/ttnn-perf/train/gpt2/default.yaml make train-gpt2

train-gpt2: train-gpt2-sync train-gpt2-build train-gpt2-run

train-gpt2-sync:
	mkdir -p $(AUTOGRAD_DIR)/autograd
	mkdir -p $(AUTOGRAD_DIR)/utils
	cp -r nn $(AUTOGRAD_DIR)/autograd/
	cp -r utils $(AUTOGRAD_DIR)/autograd/
	cp ops.hpp tensor_group.hpp value.hpp nn.hpp $(AUTOGRAD_DIR)/autograd/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../train/gpt2/train_gpt2.cpp $(AUTOGRAD_DIR)/main.cpp

train-gpt2-build:
	cd $(BUILD_DIR) && ninja example_autograd

train-gpt2-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Train Folder: MNIST MLP ===
# Usage: TRAIN_MNIST_CONFIG=~/ttnn-perf/train/mnist/default.yaml make train-mnist

train-mnist: train-mnist-sync train-mnist-build train-mnist-run

train-mnist-sync:
	mkdir -p $(AUTOGRAD_DIR)/autograd
	mkdir -p $(AUTOGRAD_DIR)/utils
	cp -r nn $(AUTOGRAD_DIR)/autograd/
	cp -r utils $(AUTOGRAD_DIR)/autograd/
	cp ops.hpp tensor_group.hpp value.hpp nn.hpp $(AUTOGRAD_DIR)/autograd/
	cp common.hpp $(AUTOGRAD_DIR)/
	cp ../train/mnist/train_mnist.cpp $(AUTOGRAD_DIR)/main.cpp

train-mnist-build:
	cd $(BUILD_DIR) && ninja example_autograd

train-mnist-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Test: ROW_MAJOR subtract behavior ===
# Checks if ttnn::subtract corrupts ROW_MAJOR tensors
test-row-major: test-row-major-sync test-row-major-build test-row-major-run

test-row-major-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp tests/test_row_major_subtract.cpp $(AUTOGRAD_DIR)/main.cpp

test-row-major-build:
	cd $(BUILD_DIR) && ninja example_autograd

test-row-major-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Test: ROW_MAJOR multiply then subtract ===
test-rm-mul-sub: test-rm-mul-sub-sync test-rm-mul-sub-build test-rm-mul-sub-run

test-rm-mul-sub-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp tests/test_row_major_multiply_subtract.cpp $(AUTOGRAD_DIR)/main.cpp

test-rm-mul-sub-build:
	cd $(BUILD_DIR) && ninja example_autograd

test-rm-mul-sub-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd

# === Test: TILE_LAYOUT subtract behavior ===
test-tile-sub: test-tile-sub-sync test-tile-sub-build test-tile-sub-run

test-tile-sub-sync:
	mkdir -p $(AUTOGRAD_DIR)
	cp common.hpp $(AUTOGRAD_DIR)/
	cp tests/test_tile_subtract.cpp $(AUTOGRAD_DIR)/main.cpp

test-tile-sub-build:
	cd $(BUILD_DIR) && ninja example_autograd

test-tile-sub-run:
	cd $(TT_METAL) && ./build_Release/ttnn/examples/example_autograd
