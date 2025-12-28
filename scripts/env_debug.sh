#!/usr/bin/env bash
set -euo pipefail

# Rebuild tt-metal with debug configuration
echo "[env_debug] Building tt-metal with --debug (Tracy enabled by default)..."
cd /home/howard/tt-metal
./build_metal.sh --debug

# Activate environment
source ~/.tenstorrent-venv/bin/activate
export TT_METAL_HOME=/home/howard/tt-metal
export ARCH_NAME=wormhole_b0
export PYTHONPATH=/home/howard/tt-metal/ttnn:/home/howard/tt-metal:/home/howard/tt-metal/tools

echo "[env_debug] Debug build complete"
echo "[env_debug] TT_METAL_HOME=$TT_METAL_HOME"
echo "[env_debug] ARCH_NAME=$ARCH_NAME"
echo "[env_debug] Tracy profiler: ENABLED"
