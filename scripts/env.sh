#!/usr/bin/env bash
set -euo pipefail

source ~/.tenstorrent-venv/bin/activate
export TT_METAL_HOME=/home/howard/tt-metal
export ARCH_NAME=wormhole_b0
export PYTHONPATH=/home/howard/tt-metal/ttnn:/home/howard/tt-metal:/home/howard/tt-metal/tools

echo "[env] Activated .tenstorrent-venv"
echo "[env] TT_METAL_HOME=$TT_METAL_HOME"
echo "[env] ARCH_NAME=$ARCH_NAME"
echo "[env] PYTHONPATH=$PYTHONPATH"
