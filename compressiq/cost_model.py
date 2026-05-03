"""Bandwidth-optimal ring all-reduce cost model (Patarasuk & Yuan, 2009).

For N workers, gradient size G bits, per-worker outbound bandwidth B_i bits/s,
per-hop latency alpha_i seconds, and per-worker compression ratio r_i in (0, 1]:

    T_i = 2(N-1)/N * (r_i * G / B_i)  +  2(N-1) * alpha_i

The ring's per-round wall clock is the bottleneck:

    T_round = max_i T_i
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class NetworkProfile:
    bandwidth_bps: float   # bits per second
    latency_s: float       # seconds per hop


def per_worker_time(
    ratios: Sequence[float],
    grad_bits: float,
    profiles: Sequence[NetworkProfile],
) -> np.ndarray:
    """Vector of per-worker ring all-reduce times for one round."""
    n = len(profiles)
    assert len(ratios) == n
    bw_term_coeff = 2.0 * (n - 1) / n
    lat_term = 2.0 * (n - 1)
    times = np.empty(n)
    for i, (r, p) in enumerate(zip(ratios, profiles)):
        times[i] = bw_term_coeff * (r * grad_bits / p.bandwidth_bps) + lat_term * p.latency_s
    return times


def round_time(
    ratios: Sequence[float],
    grad_bits: float,
    profiles: Sequence[NetworkProfile],
) -> float:
    """The bottleneck (slowest) worker time governs the ring."""
    return float(per_worker_time(ratios, grad_bits, profiles).max())


def per_worker_time_per_layer(
    ratios: np.ndarray,           # shape (N, L)
    layer_bits: Sequence[float],  # shape (L,)
    profiles: Sequence[NetworkProfile],
) -> np.ndarray:
    """Per-worker ring time when each worker sends sum_l r_{i,l} * G_l bits.

    Same ring formula; just the per-worker payload size is layer-weighted:
        T_i = 2(N-1)/N * (sum_l r_{i,l} * G_l) / B_i + 2(N-1) * alpha_i
    """
    ratios = np.asarray(ratios, dtype=float)
    layer_bits = np.asarray(layer_bits, dtype=float)
    n, L = ratios.shape
    assert len(profiles) == n
    assert layer_bits.shape == (L,)
    bw_term_coeff = 2.0 * (n - 1) / n
    lat_term = 2.0 * (n - 1)
    total_bits = ratios @ layer_bits  # (N,)
    times = np.empty(n)
    for i, p in enumerate(profiles):
        times[i] = bw_term_coeff * (total_bits[i] / p.bandwidth_bps) + lat_term * p.latency_s
    return times
