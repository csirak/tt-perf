# Tracy Device Profiler Parser

Parse Tracy device profiler logs to extract parallel kernel execution times.

## Background

When profiling TTNN ops, the device profiler outputs raw cycle data for each core and RISC processor. Since kernels execute **in parallel** across cores, simply summing durations gives incorrect results.

This parser calculates the actual wall-clock time by finding the timeline bounds (earliest kernel start to latest kernel end).

## RISC Processors

Each Tensix core has 5 RISC processors that run concurrently:

| RISC | Role | Typical Task |
|------|------|--------------|
| **BRISC** | Data movement | Read data from DRAM → L1 |
| **NCRISC** | Data movement | Write data from L1 → DRAM |
| **TRISC0/1/2** | Compute | FP math operations |

## Usage

```bash
# Use default path (~/tt-metal/generated/profiler/.logs/profile_log_device.csv)
python parse_device_log.py

# Specify custom path
python parse_device_log.py /path/to/profile_log_device.csv
```

## Generating Profile Data

```bash
cd ~/tt-metal
source python_env/bin/activate

# Run with device profiler enabled
TT_METAL_DEVICE_PROFILER=1 python your_script.py

# Profile data will be at:
# ~/tt-metal/generated/profiler/.logs/profile_log_device.csv
```

## Example Output

```
=== Device Profiler Analysis ===
ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 64

Found 6 runs

Run 6144: 48 cores
  Total parallel time: 29.80 µs
  BRISC : |========================================| 29.79 µs
  NCRISC: |==============================          | 21.48 µs
  TRISC : |================================        | 23.17 µs

=== Last Run Summary ===
Total parallel execution: 29.80 µs
Bottleneck: BRISC (29.79 µs)
Status: Memory-bound (data movement > compute)
```

## Interpreting Results

- **Memory-bound**: BRISC or NCRISC is the bottleneck → optimize data layout or reduce transfers
- **Compute-bound**: TRISC is the bottleneck → optimize math operations or use lower precision
