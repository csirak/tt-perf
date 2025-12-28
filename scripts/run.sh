#!/usr/bin/env bash
set -euo pipefail

if [ ! -f "$HOME/.tenstorrent-venv/bin/activate" ]; then
  echo "[run] Missing venv: $HOME/.tenstorrent-venv/bin/activate" >&2
  exit 1
fi

source "$HOME/.tenstorrent-venv/bin/activate"
export TT_METAL_HOME=/home/howard/tt-metal
export ARCH_NAME=wormhole_b0
export PYTHONPATH=/home/howard/tt-metal/ttnn:/home/howard/tt-metal:/home/howard/tt-metal/tools

if [ "$#" -eq 0 ]; then
  echo "[run] Usage: ./run.sh python3 your_script.py [args...]" >&2
  exit 1
fi

exec "$@"
