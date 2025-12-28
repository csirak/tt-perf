import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import PrimeSequenceDataset


def test_prime_sequence_dataset_shapes_and_shift():
    dataset = PrimeSequenceDataset(max_n=200, seq_len=32, seed=0)
    x, y = dataset.sample_batch(batch_size=32)

    assert x.shape == (32, 32)
    assert y.shape == (32, 32)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_prime_sequence_dataset_values_are_prime():
    dataset = PrimeSequenceDataset(max_n=200, seq_len=32, seed=1)
    prime_set = set(dataset.primes)
    x, y = dataset.sample_batch(batch_size=32)
    assert all(int(v) in prime_set for v in x.flatten().tolist())
    assert all(int(v) in prime_set for v in y.flatten().tolist())


def test_prime_dataset_vocab_multiple_of_32():
    dataset = PrimeSequenceDataset(max_n=1000, seq_len=32, seed=2)
    assert dataset.vocab_size % 32 == 0


if __name__ == "__main__":
    test_prime_sequence_dataset_shapes_and_shift()
    test_prime_sequence_dataset_values_are_prime()
    test_prime_dataset_vocab_multiple_of_32()
