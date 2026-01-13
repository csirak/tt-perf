#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

MAGIC = b"TDF1"
VERSION = 1
ORIG_MAGIC = b"ORIG"
OP_MAGIC = b"OPNM"

DTYPE_BF16 = 0
DTYPE_F32 = 1
DTYPE_I32 = 2
DTYPE_U32 = 3


def _dtype_from_id(dtype_id: int) -> torch.dtype:
    if dtype_id == DTYPE_BF16:
        return torch.bfloat16
    if dtype_id == DTYPE_F32:
        return torch.float32
    if dtype_id == DTYPE_I32:
        return torch.int32
    if dtype_id == DTYPE_U32:
        return torch.uint32
    raise ValueError(f"Unsupported dtype id: {dtype_id}")


def _dtype_to_id(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return DTYPE_BF16
    if dtype == torch.float32:
        return DTYPE_F32
    if dtype == torch.int32:
        return DTYPE_I32
    if dtype == torch.uint32:
        return DTYPE_U32
    raise ValueError(f"Unsupported dtype: {dtype}")


def save_tensor(t: torch.Tensor,
                path: str,
                origin: Optional[str] = None,
                op_name: Optional[str] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t = t.detach().cpu().contiguous()
    dtype_id = _dtype_to_id(t.dtype)

    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", dtype_id))
        f.write(struct.pack("<I", t.ndim))
        for dim in t.shape:
            f.write(struct.pack("<I", dim))

        if t.dtype == torch.bfloat16:
            data = t.view(torch.int16).numpy().view(np.uint16).tobytes()
        else:
            data = t.numpy().tobytes()
        f.write(data)

        if origin:
            encoded = origin.encode("utf-8")
            f.write(ORIG_MAGIC)
            f.write(struct.pack("<I", len(encoded)))
            f.write(encoded)
        if op_name:
            encoded = op_name.encode("utf-8")
            f.write(OP_MAGIC)
            f.write(struct.pack("<I", len(encoded)))
            f.write(encoded)


def load_tensor_with_meta(path: str) -> Tuple[torch.Tensor, Optional[str], Optional[str]]:
    path = Path(path)
    buf = path.read_bytes()
    if len(buf) < 4:
        raise ValueError("Tensor file is too small")

    offset = 0
    magic = buf[0:4]
    offset += 4

    if magic != MAGIC:
        # Legacy BF16 format: [ndim][dims][data]
        ndim = struct.unpack_from("<I", buf, 0)[0]
        offset = 4
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack_from("<I", buf, offset)[0])
            offset += 4
        numel = 1
        for d in shape:
            numel *= d
        data = buf[offset:offset + numel * 2]
        offset += numel * 2
        arr = np.frombuffer(data, dtype=np.uint16).astype(np.int16)
        t = torch.from_numpy(arr.copy()).view(torch.bfloat16).reshape(shape)
        origin = None
    else:
        version = struct.unpack_from("<I", buf, offset)[0]
        offset += 4
        if version != VERSION:
            raise ValueError(f"Unsupported tensordiff version: {version}")

        dtype_id = struct.unpack_from("<I", buf, offset)[0]
        offset += 4
        ndim = struct.unpack_from("<I", buf, offset)[0]
        offset += 4
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack_from("<I", buf, offset)[0])
            offset += 4
        dtype = _dtype_from_id(dtype_id)

        numel = 1
        for d in shape:
            numel *= d

        if dtype == torch.bfloat16:
            data = buf[offset:offset + numel * 2]
            offset += numel * 2
            arr = np.frombuffer(data, dtype=np.uint16).astype(np.int16)
            t = torch.from_numpy(arr.copy()).view(torch.bfloat16).reshape(shape)
        else:
            if dtype == torch.float32:
                np_dtype = np.float32
            elif dtype == torch.int32:
                np_dtype = np.int32
            elif dtype == torch.uint32:
                np_dtype = np.uint32
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
            nbytes = numel * np.dtype(np_dtype).itemsize
            raw = buf[offset:offset + nbytes]
            offset += nbytes
            arr = np.frombuffer(raw, dtype=np_dtype)
            t = torch.from_numpy(arr.copy()).reshape(shape)
            if t.dtype != dtype:
                t = t.to(dtype)
        origin = None

    op_name = None
    while offset + 8 <= len(buf):
        tag = buf[offset:offset + 4]
        tag_len = struct.unpack_from("<I", buf, offset + 4)[0]
        start = offset + 8
        end = start + tag_len
        if end > len(buf):
            break
        payload = buf[start:end].decode("utf-8", errors="replace")
        if tag == ORIG_MAGIC:
            origin = payload
        elif tag == OP_MAGIC:
            op_name = payload
        else:
            break
        offset = end

    return t, origin, op_name


def load_tensor_with_origin(path: str) -> Tuple[torch.Tensor, Optional[str]]:
    t, origin, _ = load_tensor_with_meta(path)
    return t, origin


def load_tensor(path: str) -> torch.Tensor:
    return load_tensor_with_origin(path)[0]


if __name__ == "__main__":
    # Basic round-trip test
    x = torch.randn(4, 4, dtype=torch.bfloat16)
    save_tensor(x, "/tmp/td_io_test.bin")
    y = load_tensor("/tmp/td_io_test.bin")
    assert torch.equal(x, y)
    print("td_io round-trip ok")
