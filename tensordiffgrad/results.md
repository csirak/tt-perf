# tensordiffgrad results

Log of decisions and outcomes to avoid repeating mistakes.

## 2026-01-13
- Initialized `tensordiffgrad` to use tensordiff's TDF1 recording (`record_op_cpu`) for op logging and viewer compatibility.
- Default computation dtype: f32 for both forward and backward to simplify CPU reference parity.
- Fixed torch causal mask bug (used zeros -> no masking). Switched to `torch.triu(torch.ones(...), diagonal=1) * -1e4`, which brought forward/backward parity to ~1e-6.
- Verified transformer layer parity vs torch (CPU) for RMSNorm, LayerNorm, and DyT. Max abs diffs ~1e-5 or lower; rel_l2 ~1e-7 (small ref norms cause occasional >1 rel_l2 for near-zero tensors).
- Added C++ backend ops test (`backend_ops_test.cpp`) that can be built for TTNN or Torch to generate TDF1 op logs with the same recorder.
