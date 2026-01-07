# Typecast Investigation

## Problem

Tracy profiling revealed that `ttnn::typecast` operations consume **41.1% of backward pass time** when training with BFP8 precision.

```
Operation                    Time (ms)    Percent
------------------------------------------------------------
typecast_dout_bf16             1574.44      41.1%
d_weight_compute               1621.71      42.3%
accumulate_grad_add             218.09       5.7%
...
```

This folder contains benchmarks to understand **why** typecast is so slow.

## Results (2026-01-06)

### Isolated Typecast Benchmark

| Shape | Direction | Time(ms) | GB/s | DRAM% |
|-------|-----------|----------|------|-------|
| [1,1,32,32] | BFP8→BF16 | 0.054 | 0.1 | 0.0% |
| [1,1,128,128] | BFP8→BF16 | 0.053 | 0.9 | 0.3% |
| [1,1,512,512] | BFP8→BF16 | 0.057 | 13.7 | 4.8% |
| [1,1,1024,1024] | BFP8→BF16 | 0.059 | 53.2 | 18.5% |
| [1,1,2048,2048] | BFP8→BF16 | 0.086 | 147.0 | 51.0% |
| [32,1,256,512] | BFP8→BF16 | 0.084 | 149.0 | 51.8% |
| [1,1,32,32] | BF16→BFP8 | 0.035 | 0.1 | 0.0% |
| [1,1,128,128] | BF16→BFP8 | 0.043 | 1.1 | 0.4% |
| [1,1,512,512] | BF16→BFP8 | 0.061 | 12.9 | 4.5% |
| [1,1,1024,1024] | BF16→BFP8 | 0.058 | 53.9 | 18.7% |
| [1,1,2048,2048] | BF16→BFP8 | 0.076 | 165.9 | 57.6% |
| [32,1,256,512] | BF16→BFP8 | 0.075 | 167.6 | 58.2% |

### Key Findings

1. **Fixed overhead ~50-60us**: Small shapes (32x32, 128x128) take same time as larger shapes
   - This suggests kernel launch + compile overhead dominates for small tensors
   - **This is why GPT-2 training is slow**: Many small typecast operations

2. **Peak ~58% DRAM utilization**: For large tensors (4M elements)
   - Achieved: 167 GB/s out of 288 GB/s peak
   - Room for optimization but not terrible

3. **GPT-2 activation shape [32,1,256,512]**: 4M elements
   - BFP8→BF16: 0.084 ms per typecast
   - If we have ~20 typecasts per training step: 20 * 0.084 = 1.68 ms
   - But Tracy shows 1574 ms total → **something else is going on**

### The Real Problem

The 50-60us overhead per typecast call adds up when you have many calls:
- If backward pass does 1000+ typecast operations
- 1000 * 0.05ms = 50ms just in overhead

But Tracy shows 1574ms for typecast - this suggests:
1. Much larger tensors being typecast (activation caching)
2. Or many more typecasts than expected
3. Or typecast includes memory allocation time

## Usage

```bash
cd ~/ttnn-perf/typecast
make typecast  # sync + build + run
```

## Reference Code

Local `typecast/` folder contains tt-metal source:
- `cpp/kernels/eltwise_typecast.cpp` - Compute kernel
- `llk/wormhole_b0/ckernel_sfpu_typecast.h` - LLK implementation

## Notes

- BFP8 = 1 byte per element (block floating point)
- BF16 = 2 bytes per element
- For BFP8→BF16: read 1 byte, write 2 bytes = 3 bytes/element
- For BF16→BFP8: read 2 bytes, write 1 byte = 3 bytes/element
- Peak: 288 GB/s DRAM bandwidth

## Next Steps

1. **Profile larger tensors** matching actual GPT-2 full forward activation sizes
2. **Count typecasts** in backward pass - how many are there really?
3. **Check if allocation is included** - is typecast allocating output tensor each time?
4. **Consider in-place typecast** - reuse buffers where possible
