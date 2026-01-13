# tensordiff

Minimal tensor comparison utilities to bridge TTNN tensors and Torch.

## Structure
- `core/`: CPU tensor, dtype registry, serialization, comparison helpers.
  - `ttnn_bridge.hpp`: TTNN conversion helpers.
  - `torch_bridge.hpp`: libtorch conversion helpers.
- `ttnn_tensor.hpp`: TTNN-backed tensor implementation.
- `torch_tensor.hpp`: libtorch-backed tensor implementation.
- `compare_ttnn_torch.cpp`: TTNN-side round-trip + binary dump.
- `compare_ttnn_torch.py`: Torch-side loader + comparison.
- `tests/tensor_ops_test.cpp`: shared ops test (TTNN default, Torch when compiled with `TENSORDIFF_TEST_TORCH`).

## Current state
- Serialization header supports BF16/F32/I32/U32; current flows emit BF16.
- CPU tensor supports BF16/F32; TTNN bridge currently emits BF16.
- Libtorch bridge uses CPU tensors + float conversion (safe, not zero-copy).
- BaseTensor ops (`add/sub/mm`) can auto-record inputs/outputs and an op manifest.

## Serialization format (TDF1)
All tensordiff `.bin` files use the same header:
- Magic: `TDF1`
- Version: u32 (1)
- DType: u32 (0=bf16, 1=f32, 2=i32, 3=u32)
- NDim: u32
- Dims: u32 * ndim
- Data: raw bytes

Optional trailer (backward compatible):
- `ORIG` + u32 length + UTF-8 origin string (e.g., `ttnn`, `torch`, `cpu`, `cuda`)
- `OPNM` + u32 length + UTF-8 op name (e.g., `mm`, `softmax_bw`)

### Python save example (Torch)
```python
from tensordiff.td_io import save_tensor, load_tensor_with_origin
import torch

x = torch.randn(32, 32, dtype=torch.bfloat16)
save_tensor(x, "tensordiff/outputs/torch/example.bin", origin="torch")

t, origin = load_tensor_with_origin("tensordiff/outputs/torch/example.bin")
print(origin)
```

## Op recording (automatic)
Enable a single recorder for all `BaseTensor` ops and it will emit:
- Per-op tensor files (`NNNN_<op>_in*.bin`, `NNNN_<op>_out.bin`)
- A manifest `ops.tsv` with input/output paths + shapes + origins + opts
- `OPNM` metadata on output tensors (shown in the viewer)

```cpp
#include "core/base_tensor.hpp"

using tensordiff::core::enable_op_recording;
using tensordiff::core::disable_op_recording;

enable_op_recording("tensordiff/outputs/oplog");
auto out = a->mm(*b);  // records op + inputs/outputs
disable_op_recording();
```

Manifest columns:
```
idx	op	type	inputs	input_shapes	input_origins	input_dtypes	output	output_shape	output_origin	output_dtype	opts
```

### Compile-time switch
To remove recording overhead entirely, build with:
```
-DTENSORDIFF_ENABLE_RECORDING=0
```
When disabled, `enable_op_recording()` and all record helpers are no-ops.

### Recording from CPU tensors (for autograd ops)
You can log any op (including ones outside `BaseTensor`) using CPU tensors:
```cpp
using tensordiff::core::record_op_cpu;
using tensordiff::core::OpType;

std::vector<const CpuTensor*> inputs = {&in0, &in1};
std::vector<std::string_view> origins = {"ttnn", "ttnn"};
record_op_cpu("softmax_bw", OpType::Binary, inputs, origins, out, "ttnn", "axis=2");
```

## Usage
1) Run TTNN writer (build via autograd Makefile target):
```
ssh howard "cd ~/ttnn-perf/autograd && make tensordiff-compare"
```

2) Compare in torch:
```
ssh howard "cd ~/ttnn-perf && ./scripts/run.sh python3 tensordiff/compare_ttnn_torch.py --dir tensordiff/outputs/ttnn"
```

3) Compare in libtorch (C++):
```
ssh howard "cd ~/ttnn-perf/autograd && make tensordiff-torch-compare"
```

4) Render heatmap of diffs (2D tensors):
```
ssh howard "cd ~/ttnn-perf && ./scripts/run.sh python3 tensordiff/heatmap_diff.py \\
  --a tensordiff/outputs/ttnn/cpu_ref.bin \\
  --b tensordiff/outputs/ttnn/ttnn_roundtrip.bin \\
  --out tensordiff/outputs/ttnn/diff_heatmap.png"
```

5) Tensor ops test (TTNN + Torch):
```
ssh howard "cd ~/ttnn-perf/autograd && make tensordiff-ops-test"
ssh howard "cd ~/ttnn-perf/autograd && make tensordiff-ops-test-torch"
```

Outputs:
- `tensordiff/outputs/ttnn/cpu_ref.bin`
- `tensordiff/outputs/ttnn/ttnn_roundtrip.bin`
