"""MNIST loading and IID sharding across N simulated workers."""
from __future__ import annotations

import os
from typing import List

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_mnist(data_root: str = "./data") -> tuple[datasets.MNIST, datasets.MNIST]:
    os.makedirs(data_root, exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
    return train, test


def shard_dataset(dataset, num_workers: int, seed: int = 0) -> List[Subset]:
    """Split a dataset into `num_workers` IID shards."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    shard_size = len(dataset) // num_workers
    shards = []
    for i in range(num_workers):
        idx = perm[i * shard_size : (i + 1) * shard_size]
        shards.append(Subset(dataset, idx))
    return shards


def make_loaders(shards: List[Subset], batch_size: int = 64) -> List[DataLoader]:
    return [DataLoader(s, batch_size=batch_size, shuffle=True) for s in shards]
