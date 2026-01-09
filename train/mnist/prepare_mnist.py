#!/usr/bin/env python3
"""Prepare MNIST dataset without torchvision.

Downloads raw IDX files, parses them, pads to multiple-of-32 input dim,
then saves BF16 features and uint32 labels in a simple binary format.
"""

import argparse
import gzip
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
sys.path.append(str(REPO_ROOT / "experiments" / "grok"))
from tensor_io import save_tensor

MNIST_BASES = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
}


def load_kv_file(path: Path) -> dict:
    kv = {}
    if not path.exists():
        return kv
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, val = line.split(":", 1)
        kv[key.strip()] = val.strip()
    return kv


def get_int(kv: dict, key: str, default: int) -> int:
    return int(kv.get(key, default))


def get_float(kv: dict, key: str, default: float) -> float:
    return float(kv.get(key, default))


def get_str(kv: dict, key: str, default: str) -> str:
    return str(kv.get(key, default))


def download(urls: list[str], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    last_err = None
    for url in urls:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        try:
            print(f"Downloading {url} -> {dest}")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            return
        except Exception as err:  # pylint: disable=broad-except
            last_err = err
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"Failed to download {dest.name}: {last_err}")


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad MNIST image magic: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != num * rows * cols:
        raise ValueError("MNIST image file size mismatch")
    return data.reshape(num, rows, cols)


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad MNIST label magic: {magic}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != num:
        raise ValueError("MNIST label file size mismatch")
    return data


def save_u32_tensor(arr: np.ndarray, path: Path) -> None:
    arr = np.asarray(arr, dtype=np.uint32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<I", arr.ndim))
        for dim in arr.shape:
            f.write(struct.pack("<I", dim))
        f.write(arr.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MNIST without torchvision")
    parser.add_argument("--config", type=str, default=str(THIS_DIR / "default.yaml"))
    parser.add_argument("--data-dir", type=str, default=None)
    args = parser.parse_args()

    kv = load_kv_file(Path(args.config))
    data_dir = Path(args.data_dir or get_str(kv, "data_dir", str(THIS_DIR / "data")))
    train_samples = get_int(kv, "train_samples", 256)
    seed = get_int(kv, "seed", 1)
    pad_to = get_int(kv, "pad_to", 1024)
    num_classes = get_int(kv, "num_classes", 32)

    data_dir.mkdir(parents=True, exist_ok=True)

    img_path = data_dir / FILES["train_images"]
    lbl_path = data_dir / FILES["train_labels"]

    img_urls = [base + FILES["train_images"] for base in MNIST_BASES]
    lbl_urls = [base + FILES["train_labels"] for base in MNIST_BASES]
    download(img_urls, img_path)
    download(lbl_urls, lbl_path)

    images = read_idx_images(img_path)
    labels = read_idx_labels(lbl_path)

    if images.shape[0] != labels.shape[0]:
        raise ValueError("MNIST images/labels size mismatch")

    total = images.shape[0]
    if train_samples > total:
        raise ValueError(f"train_samples ({train_samples}) > dataset size ({total})")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)[:train_samples]

    images = images[indices]
    labels = labels[indices]

    flat = images.reshape(train_samples, -1).astype(np.float32) / 255.0
    if flat.shape[1] > pad_to:
        raise ValueError(f"pad_to ({pad_to}) smaller than flat dim ({flat.shape[1]})")

    padded = np.zeros((train_samples, pad_to), dtype=np.float32)
    padded[:, : flat.shape[1]] = flat

    x_path = data_dir / "train_images.bin"
    y_path = data_dir / "train_labels_u32.bin"

    save_tensor(torch.from_numpy(padded), str(x_path))
    save_u32_tensor(labels.astype(np.uint32), y_path)

    meta = {
        "train_samples": train_samples,
        "pad_to": pad_to,
        "num_classes": num_classes,
        "seed": seed,
        "images_path": str(x_path),
        "labels_path": str(y_path),
        "raw_images": str(img_path),
        "raw_labels": str(lbl_path),
    }
    meta_path = data_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved: {x_path}")
    print(f"Saved: {y_path}")
    print(f"Meta:  {meta_path}")


if __name__ == "__main__":
    main()
