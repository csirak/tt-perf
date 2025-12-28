import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import ModularDivisionDataset


def test_modular_division_shapes_and_tokens():
    dataset = ModularDivisionDataset(p=97, train_frac=0.5, pad_to=32, seed=0)
    x, y = dataset.sample_batch(batch_size=32, split="train")

    assert x.shape == (32, 32)
    assert y.shape == (32, 1)
    assert dataset.base_vocab_size == 99
    assert dataset.vocab_size % 32 == 0

    assert torch.all((x[:, 0] >= 0) & (x[:, 0] < 97))
    assert torch.all(x[:, 1] == dataset.div_token)
    assert torch.all((x[:, 2] >= 1) & (x[:, 2] < 97))
    assert torch.all(x[:, 3] == dataset.eq_token)
    assert torch.all(x[:, 4:] == 0)


def test_modular_division_labels_correct():
    dataset = ModularDivisionDataset(p=97, train_frac=0.5, pad_to=32, seed=1)
    tokens, targets = dataset.sample_batch(batch_size=32, split="train")
    x = tokens[:, 0].tolist()
    y = tokens[:, 2].tolist()
    z = targets[:, 0].tolist()
    for xi, yi, zi in zip(x, y, z):
        assert (xi * pow(yi, 95, 97)) % 97 == zi


def test_modular_division_split_sizes():
    dataset = ModularDivisionDataset(p=97, train_frac=0.5, pad_to=32, seed=2)
    total = 97 * 96
    assert dataset.train_size + dataset.val_size == total
    assert dataset.train_size == dataset.val_size


if __name__ == "__main__":
    test_modular_division_shapes_and_tokens()
    test_modular_division_labels_correct()
    test_modular_division_split_sizes()
