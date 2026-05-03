"""Convex optimization for per-worker compression ratios + baselines.

Problem (CompressIQ):

    minimize    max_i  r_i * G / B_i
    subject to  sum_i  kappa_i * alpha_i * (1 - r_i)^2  <=  epsilon
                r_min <= r_i <= 1

The objective is the max of linear functions (convex); the accuracy constraint
is a sum of convex quadratics. Solved with CVXPY (ECOS).

The 2(N-1)/N bandwidth coefficient and 2(N-1)*alpha latency term are constants
in r_i, so they drop out of argmin. We still report them in the cost model used
for simulation wall-clock measurement.

`kappa_i` is an EF-aware correction factor (defaults to 1): under error
feedback, the per-round leaked squared error is consistently smaller than the
naive bound `(1-r_i)^2 * ||g_i||^2` because past residuals are folded in. Set
it to the empirical ratio emp_leaked / theoretical_bound (EMA across rounds)
to get a tighter constraint. See Karimireddy et al. 2019.

Per-layer variant (`solve_compressiq_per_layer`) extends this to a matrix of
ratios R[i, l], with per-layer gradient sizes G_l and per-worker-per-layer
accuracy weights alpha[i, l]. Keeps the exact same convex structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cvxpy as cp
import numpy as np

from .cost_model import NetworkProfile


@dataclass
class SolveResult:
    ratios: np.ndarray
    objective_value: float
    status: str


def solve_compressiq(
    profiles: Sequence[NetworkProfile],
    grad_bits: float,
    alphas: Sequence[float],
    epsilon: float,
    r_min: float = 0.01,
    ef_factors: Optional[Sequence[float]] = None,
) -> SolveResult:
    """Per-worker optimal compression ratios under accuracy budget.

    `ef_factors` (kappa_i) is an optional per-worker multiplier on the accuracy
    constraint. Pass the empirical ratio emp_leaked / theoretical_bound (EMA)
    to get an EF-aware bound. Defaults to all 1s (naive bound).
    """
    n = len(profiles)
    bw = np.array([p.bandwidth_bps for p in profiles])
    alphas = np.asarray(alphas, dtype=float)
    if ef_factors is None:
        kappa = np.ones(n)
    else:
        kappa = np.clip(np.asarray(ef_factors, dtype=float), 1e-6, 1.0)

    r = cp.Variable(n)
    # Objective is invariant to positive scaling; we minimize max_i r_i / B_i,
    # but rescale by min(bw) so coefficients are O(1) and the solver doesn't
    # see an all-tiny-values objective (which collapses to numerical noise).
    coeffs = bw.min() / bw  # in (0, 1]
    objective = cp.Minimize(cp.max(cp.multiply(coeffs, r)))
    constraints = [
        r >= r_min,
        r <= 1.0,
        cp.sum(cp.multiply(kappa * alphas, cp.square(1.0 - r))) <= epsilon,
    ]
    prob = cp.Problem(objective, constraints)
    # Try a sequence of solvers depending on what's installed in the env.
    installed = set(cp.installed_solvers())
    for solver in ("CLARABEL", "ECOS", "SCS"):
        if solver in installed:
            prob.solve(solver=solver)
            break
    else:
        prob.solve()
    if r.value is None:
        # Fallback: infeasible budget -> uniform at r_min
        return SolveResult(ratios=np.full(n, r_min), objective_value=float("inf"), status=prob.status)
    ratios = np.clip(r.value, r_min, 1.0)
    return SolveResult(ratios=ratios, objective_value=float(prob.value), status=prob.status)


def solve_compressiq_per_layer(
    profiles: Sequence[NetworkProfile],
    layer_bits: Sequence[float],
    alphas: np.ndarray,  # shape (N, L): per-worker per-layer ||g||^2
    epsilon: float,
    r_min: float = 0.01,
    ef_factors: Optional[np.ndarray] = None,  # shape (N, L) or (N,) or None
) -> SolveResult:
    """Per-worker **per-layer** optimal compression ratios.

    Decision variable is R with shape (N, L). Objective matches the ring
    all-reduce bottleneck (each worker sends sum_l R[i,l]*G_l bits).

        minimize    max_i  sum_l R[i,l] * G_l / B_i
        subject to  sum_{i,l} kappa[i,l] * alpha[i,l] * (1 - R[i,l])^2  <= eps
                    r_min <= R[i,l] <= 1

    Still one SOC constraint + box bounds; still convex; still <10 ms for
    realistic (N, L).
    """
    alphas = np.asarray(alphas, dtype=float)
    assert alphas.ndim == 2, f"alphas must be (N, L); got shape {alphas.shape}"
    n, L = alphas.shape
    assert len(profiles) == n
    layer_bits = np.asarray(layer_bits, dtype=float)
    assert layer_bits.shape == (L,)
    bw = np.array([p.bandwidth_bps for p in profiles])

    if ef_factors is None:
        kappa = np.ones_like(alphas)
    else:
        k = np.asarray(ef_factors, dtype=float)
        if k.ndim == 1:
            k = np.broadcast_to(k[:, None], (n, L)).copy()
        kappa = np.clip(k, 1e-6, 1.0)

    R = cp.Variable((n, L))
    # Each worker sends sum_l R[i,l] * G_l bits total; rescale by B_min for
    # numerical conditioning same as scalar case.
    bw_min = float(bw.min())
    # Per-worker total compressed bits (linear in R), scaled by B_min/B_i.
    per_worker_bits = R @ layer_bits  # shape (n,)
    coeffs = bw_min / bw
    objective = cp.Minimize(cp.max(cp.multiply(coeffs, per_worker_bits)))

    constraints = [
        R >= r_min,
        R <= 1.0,
        cp.sum(cp.multiply(kappa * alphas, cp.square(1.0 - R))) <= epsilon,
    ]
    prob = cp.Problem(objective, constraints)
    installed = set(cp.installed_solvers())
    for solver in ("CLARABEL", "ECOS", "SCS"):
        if solver in installed:
            prob.solve(solver=solver)
            break
    else:
        prob.solve()
    if R.value is None:
        return SolveResult(ratios=np.full((n, L), r_min),
                           objective_value=float("inf"), status=prob.status)
    ratios = np.clip(R.value, r_min, 1.0)
    return SolveResult(ratios=ratios, objective_value=float(prob.value), status=prob.status)


def baseline_none(n: int) -> np.ndarray:
    return np.ones(n)


def baseline_uniform(
    n: int,
    alphas: Sequence[float],
    epsilon: float,
    r_min: float = 0.01,
) -> np.ndarray:
    """Single ratio r* applied to all workers, satisfying the accuracy bound.

    sum_i alpha_i (1 - r)^2 <= epsilon
    => (1 - r)^2 <= epsilon / sum(alpha_i)
    => r >= 1 - sqrt(epsilon / sum(alpha_i))
    """
    alpha_sum = float(np.sum(alphas))
    r_star = 1.0 - np.sqrt(max(epsilon, 0.0) / max(alpha_sum, 1e-12))
    r_star = float(np.clip(r_star, r_min, 1.0))
    return np.full(n, r_star)


def baseline_greedy(
    profiles: Sequence[NetworkProfile],
    alphas: Sequence[float],
    epsilon: float,
    r_min: float = 0.01,
) -> np.ndarray:
    """Each worker independently picks the smallest r_i it can afford using a
    per-worker share of the budget proportional to (B_max / B_i) -- i.e. slow
    workers get more budget. Does not coordinate across workers."""
    n = len(profiles)
    bw = np.array([p.bandwidth_bps for p in profiles])
    weights = (bw.max() / bw)  # slow workers -> larger weight -> larger share
    shares = epsilon * weights / weights.sum()
    ratios = []
    for i in range(n):
        # alpha_i * (1 - r_i)^2 <= shares_i  =>  r_i >= 1 - sqrt(shares_i / alpha_i)
        r_i = 1.0 - np.sqrt(max(shares[i], 0.0) / max(alphas[i], 1e-12))
        ratios.append(float(np.clip(r_i, r_min, 1.0)))
    return np.array(ratios)
