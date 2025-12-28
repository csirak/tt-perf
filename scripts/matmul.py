#!/usr/bin/env python3
import os
import sys

DEFAULT_TT_METAL_HOME = "/home/howard/tt-metal"

if "TT_METAL_HOME" not in os.environ:
    os.environ["TT_METAL_HOME"] = DEFAULT_TT_METAL_HOME
if "ARCH_NAME" not in os.environ:
    os.environ["ARCH_NAME"] = "wormhole_b0"
if DEFAULT_TT_METAL_HOME not in sys.path:
    sys.path.insert(0, DEFAULT_TT_METAL_HOME)

import torch
import ttnn


def main() -> int:
    torch.manual_seed(0)

    a = torch.randn(1, 1, 32, 32)
    b = torch.randn(1, 1, 32, 32)

    with ttnn.manage_device(0) as device:
        a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
        b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
        c_tt = ttnn.matmul(a_tt, b_tt)
        c = ttnn.to_torch(c_tt)

    ref = torch.matmul(a, b)
    max_err = (c - ref).abs().max().item()
    print("max_abs_err", max_err)
    print("c_shape", tuple(c.shape))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
