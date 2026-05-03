"""Discrete-event simulation of distributed training with ring all-reduce.

Each round:
  1. Every worker computes a local gradient on its shard.
  2. Every worker compresses its gradient at its assigned ratio r_i.
  3. The ring all-reduce produces the averaged compressed gradient.
     Wall-clock for the round = max_i T_i(r_i) using the corrected
     bandwidth-optimal ring formula.
  4. Every worker applies the same averaged update.

Note: For simulation efficiency, we do not literally pass chunks around a
ring. Because ring all-reduce is *mathematically* equivalent to averaging
all (compressed) gradients and broadcasting, we compute the average directly
and charge the simulated clock the bottleneck time. The DES property we care
about -- that the round advances by max_i T_i, not sum_i T_i -- is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .cost_model import (
    NetworkProfile,
    per_worker_time,
    per_worker_time_per_layer,
    round_time,
)
from .worker import GPUWorker, num_params


@dataclass
class RoundLog:
    round_idx: int
    sim_time_s: float          # cumulative simulated wall clock
    round_time_s: float        # bottleneck time this round
    bottleneck_worker: int
    test_accuracy: float
    test_loss: float
    ratios: np.ndarray = field(default_factory=lambda: np.array([]))  # scalar (N,) or per-layer (N, L)
    # Diagnostics for the adaptive / bound-validation experiment
    alpha_now: np.ndarray = field(default_factory=lambda: np.array([]))   # ||g_i||^2 this round
    theoretical_error: float = 0.0   # sum_i alpha_now_i * (1 - r_i)^2  (model prediction)
    empirical_error: float = 0.0     # sum_i ||g_i - sent_i||^2 reported by worker.compress()
    kappa_ema: np.ndarray = field(default_factory=lambda: np.array([]))   # EMA of emp/theo per worker
    per_worker_emp: np.ndarray = field(default_factory=lambda: np.array([]))  # per-worker empirical sq error
    epsilon: float = float("inf")    # the budget (echoed for plotting)
    recalibrated: bool = False       # True on rounds where ratios were re-solved


def calibrate_alphas(workers: Sequence[GPUWorker]) -> np.ndarray:
    """Estimate accuracy-sensitivity alpha_i = ||g_i||^2 from one batch each."""
    alphas = []
    for w in workers:
        g = w.compute_gradient()
        alphas.append(float(g.pow(2).sum().item()))
    return np.array(alphas)


def calibrate_layer_alphas(workers: Sequence[GPUWorker]) -> np.ndarray:
    """Per-worker per-layer squared grad norms from one batch each. Shape (N, L)."""
    rows = []
    for w in workers:
        g = w.compute_gradient()
        rows.append(w.compute_layer_alphas(g))
    return np.array(rows)


def evaluate(model: torch.nn.Module, test_loader: DataLoader, device) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += float(F.cross_entropy(logits, y, reduction="sum").item())
            correct += int((logits.argmax(1) == y).sum().item())
            total += y.size(0)
    return correct / total, loss_sum / total


def run_training(
    workers: List[GPUWorker],
    ratios,  # (N,) scalar ratios OR (N, L) per-layer ratios
    grad_bits: float,
    test_loader: DataLoader,
    num_rounds: int = 200,
    lr: float = 0.05,
    eval_every: int = 10,
    device: str = "cpu",
    verbose: bool = False,
    *,
    epsilon: Optional[float] = None,
    recalibrate_every: Optional[int] = None,
    resolve_fn: Optional[Callable[..., np.ndarray]] = None,
    layer_bits_vec: Optional[Sequence[float]] = None,
    kappa_ema_rate: float = 0.2,
) -> List[RoundLog]:
    """DES training loop.

    By default `ratios` are fixed for the whole run (the static formulation).
    If `recalibrate_every` and `resolve_fn` are provided, every K rounds we
    re-measure alphas and call `resolve_fn(alphas, kappa=kappa_ema)` to obtain
    fresh ratios (the adaptive formulation). `resolve_fn` may ignore `kappa`
    if it doesn't need EF-awareness.

    Per-layer mode is entered automatically when `ratios` has 2 dimensions.
    Wall-clock timing then uses `per_worker_time_per_layer`, requiring
    `layer_bits_vec`.

    `kappa_ema_rate` is the EMA coefficient for per-worker empirical/theoretical
    leakage; 0 disables tracking.

    `epsilon` is logged per round so plots can draw the budget line; it does
    not affect dynamics by itself.
    """
    profiles = [w.profile for w in workers]
    ratios = np.asarray(ratios, dtype=float)
    per_layer = ratios.ndim == 2
    if per_layer:
        assert layer_bits_vec is not None, "layer_bits_vec required for per-layer mode"
        layer_bits_vec = np.asarray(layer_bits_vec, dtype=float)
    sim_time = 0.0
    logs: List[RoundLog] = []

    n = len(workers)
    n_params = num_params(workers[0].model)

    epsilon = float("inf") if epsilon is None else float(epsilon)
    # EMA of (per-worker empirical leakage) / (per-worker theoretical bound).
    # Initialized to 1 (conservative = naive bound).
    kappa_ema = np.ones(n)

    for rd in range(num_rounds):
        # 0. (optional) recalibrate alphas + re-solve for fresh ratios
        recalibrated = False
        if (
            recalibrate_every is not None
            and resolve_fn is not None
            and rd > 0
            and rd % recalibrate_every == 0
        ):
            if per_layer:
                alphas_new = calibrate_layer_alphas(workers)
            else:
                alphas_new = calibrate_alphas(workers)
            try:
                ratios = np.asarray(
                    resolve_fn(alphas_new, kappa=kappa_ema.copy()), dtype=float,
                )
            except TypeError:
                # resolve_fn doesn't accept kappa -> naive recall
                ratios = np.asarray(resolve_fn(alphas_new), dtype=float)
            recalibrated = True
            if verbose:
                print(f"[round {rd+1:4d}]  recalibrated. "
                      f"alphas_shape={alphas_new.shape}  kappa={np.round(kappa_ema,3)}  "
                      f"ratios_shape={ratios.shape}")

        # 1. local gradients + per-worker compression diagnostics
        compressed = torch.zeros(n_params, device=device)
        alpha_now = np.zeros(n)
        per_worker_emp = np.zeros(n)
        for i, w in enumerate(workers):
            g = w.compute_gradient()
            alpha_now[i] = float(g.pow(2).sum().item())
            r_i = ratios[i] if per_layer else float(ratios[i])
            c, leaked_sq = w.compress(g, r_i)
            compressed += c
            per_worker_emp[i] = leaked_sq
        empirical_err = float(per_worker_emp.sum())
        avg_grad = compressed / n

        # 2. per-round simulated wall clock (bottleneck)
        if per_layer:
            times = per_worker_time_per_layer(ratios, layer_bits_vec, profiles)
        else:
            times = per_worker_time(ratios, grad_bits, profiles)
        rt = float(times.max())
        bottleneck = int(times.argmax())
        sim_time += rt

        # 3. apply same update everywhere (ring all-reduce semantics)
        for w in workers:
            w.apply_gradient(avg_grad, lr)

        # 4. theoretical per-worker error using current alpha estimate
        if per_layer:
            # alphas_now is scalar per worker; for theoretical bound we need
            # per-layer alphas too -- recompute lazily only here, keeping the
            # cost proportional to n_params per round.
            per_worker_theo = np.zeros(n)
            # Re-split last g is expensive; instead reconstruct from alpha_now by
            # applying the per-layer bound using per-layer gradient norms estimated
            # in step 1. We didn't save them to avoid extra state; approximate by
            # uniformly distributing alpha_now across layers via per-layer bits.
            # This is only used for diagnostics/logging.
            layer_weights = layer_bits_vec / layer_bits_vec.sum()
            for i in range(n):
                # approximate: alpha_{i,l} ~ alpha_i * (layer_weight_l)
                per_worker_theo[i] = np.sum(
                    (alpha_now[i] * layer_weights) * (1.0 - ratios[i]) ** 2
                )
        else:
            per_worker_theo = alpha_now * (1.0 - ratios) ** 2
        theoretical_err = float(per_worker_theo.sum())

        # 4b. Update kappa EMA (per-worker empirical / theoretical leakage).
        if kappa_ema_rate > 0:
            ratio_now = per_worker_emp / np.maximum(per_worker_theo, 1e-12)
            ratio_now = np.clip(ratio_now, 1e-6, 1.0)  # bound is an upper bound
            kappa_ema = (1 - kappa_ema_rate) * kappa_ema + kappa_ema_rate * ratio_now

        # 5. periodic eval (always also log the last round)
        do_eval = ((rd + 1) % eval_every == 0) or (rd == num_rounds - 1) or recalibrated
        if do_eval:
            acc, loss = evaluate(workers[0].model, test_loader, device)
            log = RoundLog(
                round_idx=rd,
                sim_time_s=sim_time,
                round_time_s=rt,
                bottleneck_worker=bottleneck,
                test_accuracy=acc,
                test_loss=loss,
                ratios=ratios.copy(),
                alpha_now=alpha_now.copy(),
                theoretical_error=theoretical_err,
                empirical_error=empirical_err,
                kappa_ema=kappa_ema.copy(),
                per_worker_emp=per_worker_emp.copy(),
                epsilon=epsilon,
                recalibrated=recalibrated,
            )
            logs.append(log)
            if verbose:
                print(f"[round {rd+1:4d}]  sim_time={sim_time:8.3f}s  "
                      f"round={rt*1e3:7.2f}ms  bot=W{bottleneck}  acc={acc:.4f}  "
                      f"theo={theoretical_err:.4f} emp={empirical_err:.4f} "
                      f"kappa_mean={kappa_ema.mean():.3f} eps={epsilon:.4f}")
    return logs
