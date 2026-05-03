"""Helpers to build a heterogeneous simulated cluster + workers."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .cost_model import NetworkProfile
from .data import load_mnist, make_loaders, shard_dataset
from .worker import GPUWorker, MLP


def make_heterogeneous_profiles(
    n: int,
    fast_bw_gbps: float = 10.0,
    slow_bw_gbps: float = 1.0,
    fast_lat_us: float = 5.0,
    slow_lat_us: float = 50.0,
    slow_fraction: float = 0.25,
    seed: int = 0,
) -> List[NetworkProfile]:
    """Two-tier cluster: fast intra-rack vs slow inter-rack links."""
    rng = np.random.default_rng(seed)
    n_slow = max(1, int(round(slow_fraction * n)))
    is_slow = np.zeros(n, dtype=bool)
    is_slow[rng.choice(n, size=n_slow, replace=False)] = True
    profiles = []
    for i in range(n):
        if is_slow[i]:
            profiles.append(NetworkProfile(slow_bw_gbps * 1e9, slow_lat_us * 1e-6))
        else:
            profiles.append(NetworkProfile(fast_bw_gbps * 1e9, fast_lat_us * 1e-6))
    return profiles


# A "tier" is (bandwidth_gbps, latency_us, fraction_of_workers).
TierSpec = Tuple[float, float, float]


def make_tiered_profiles(
    n: int,
    tiers: Sequence[TierSpec],
    seed: int = 0,
) -> List[NetworkProfile]:
    """Multi-tier cluster. `tiers` is a list of (bw_gbps, latency_us, fraction)
    tuples whose fractions should sum to ~1.0.

    Example (three-tier datacenter: NVLink-ish / PCIe-ish / slow Ethernet):
        tiers = [(10.0, 5.0, 0.5),    # 50% fast
                 ( 5.0, 20.0, 0.25),  # 25% medium
                 ( 1.0, 50.0, 0.25)]  # 25% slow
    """
    rng = np.random.default_rng(seed)
    fractions = np.array([t[2] for t in tiers], dtype=float)
    if not np.isclose(fractions.sum(), 1.0, atol=1e-3):
        raise ValueError(f"tier fractions must sum to 1.0, got {fractions.sum():.3f}")

    # Deterministically assign counts per tier, making sure every tier gets >=1
    # worker when its fraction is nonzero.
    counts = np.maximum(np.round(fractions * n).astype(int), (fractions > 0).astype(int))
    # Fix rounding drift so counts sum to n.
    while counts.sum() > n:
        counts[np.argmax(counts)] -= 1
    while counts.sum() < n:
        counts[np.argmax(fractions)] += 1

    tier_of_worker = np.concatenate([np.full(c, k) for k, c in enumerate(counts)])
    rng.shuffle(tier_of_worker)

    profiles = []
    for i in range(n):
        bw_gbps, lat_us, _ = tiers[tier_of_worker[i]]
        profiles.append(NetworkProfile(bw_gbps * 1e9, lat_us * 1e-6))
    return profiles


def build_cluster(
    n_workers: int,
    profiles: List[NetworkProfile],
    batch_size: int = 64,
    data_root: str = "./data",
    device: str = "cpu",
    seed: int = 0,
    use_error_feedback: bool = True,
) -> tuple[List[GPUWorker], DataLoader, int]:
    """Returns (workers, test_loader, grad_bits)."""
    torch.manual_seed(seed)
    train, test = load_mnist(data_root)
    shards = shard_dataset(train, n_workers, seed=seed)
    loaders = make_loaders(shards, batch_size=batch_size)

    # All workers share the same initial weights (data-parallel training).
    base = MLP()
    init_state = {k: v.clone() for k, v in base.state_dict().items()}

    workers = []
    for i in range(n_workers):
        m = MLP()
        m.load_state_dict(init_state)
        workers.append(
            GPUWorker(
                worker_id=i,
                model=m,
                loader=loaders[i],
                profile=profiles[i],
                device=device,
                use_error_feedback=use_error_feedback,
            )
        )

    grad_bits = sum(p.numel() for p in workers[0].model.parameters()) * 32  # fp32
    test_loader = DataLoader(test, batch_size=512)
    return workers, test_loader, grad_bits
