# tensordiffgrad

CPU autograd + tensordiff recording, with a Torch CPU reference for parity.

## Build + run (CPU autograd)
From repo root:
```
make -C tensordiffgrad transformer-test OUT_DIR=outputs/run0
```

Optional env vars:
- `TENSORDIFFGRAD_NORM=layernorm|rmsnorm|dyt` (default: rmsnorm)
- `TENSORDIFFGRAD_LAYERS=<N>` (default: 1)
- `TENSORDIFF_OPLOG_DIR=/path/to/oplog` (enables per-op recording in TDF1 format)

### Op recording (TDF1)
When `TENSORDIFF_OPLOG_DIR` is set, op recording is enabled globally and automatically.
Any op that uses the tensordiff recorder will emit:
- `ops.tsv` (op log with inputs/outputs/opts)
- `NNNN_<op>_in*.bin` / `NNNN_<op>_out.bin` (TDF1 tensors)

This is the same TDF1 format used for torch ⇄ ttnn CPU interchange.

## Torch CPU reference + compare
```
./scripts/run.sh python3 tensordiffgrad/torch/transformer_layer_ref.py --dir tensordiffgrad/outputs/run0
./scripts/run.sh python3 tensordiffgrad/torch/compare_outputs.py --dir tensordiffgrad/outputs/run0
```

## C++ backend ops tests (TTNN + Torch)
TTNN backend (runs on howard via tt-metal build):
```
ssh howard "cd ~/ttnn-perf/autograd && make tensordiffgrad-ops-test"
```

Torch C++ backend (libtorch, CPU):
```
ssh howard "cd ~/ttnn-perf/autograd && make tensordiffgrad-ops-test-torch"
```

Both tests emit `ops.tsv` + per-op tensors when `TENSORDIFF_OPLOG_DIR` is set (TTNN target sets it to
`~/ttnn-perf/tensordiffgrad/outputs/oplog` and Torch target sets
`~/ttnn-perf/tensordiffgrad/outputs/oplog_torch` by default).

## FFN test (Torch CPU FP32 + BF16)
```
ssh howard "cd ~/ttnn-perf/autograd && make tensordiffgrad-ffn-test-torch"
```
Outputs:
- `~/ttnn-perf/experiments/ffn_forward/torch/torch_fp32`
- `~/ttnn-perf/experiments/ffn_forward/torch/torch_bf16`

Each contains `oplog/ops.tsv` + TDF1 tensors when `TENSORDIFF_OPLOG_DIR` is set.

## Outputs
- `outputs/run0/*.bin`: TDF1 tensors (viewer-compatible)
- `outputs/run0/torch_*.bin`: Torch reference tensors
- `ops.tsv` + `NNNN_<op>_*.bin` (if recording enabled)
