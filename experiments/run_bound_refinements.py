"""Bound-refinement ablation experiment.

Compares four CompressIQ variants at matched epsilon / recalibration:
  - naive        : scalar r_i, naive bound Sigma alpha_i (1-r_i)^2 <= eps
  - ef           : scalar r_i, EF-aware bound using empirical kappa_i
  - per_layer    : per-layer r_{i,l}, naive bound
  - per_layer_ef : per-layer r_{i,l}, EF-aware bound

Each run trains the MNIST MLP for `--rounds` rounds with periodic recalibration.

Outputs:
  results/refine_rounds.csv
  results/refine_summary.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compressiq.cluster import build_cluster, make_heterogeneous_profiles, make_tiered_profiles
from compressiq.optimizer import solve_compressiq, solve_compressiq_per_layer
from compressiq.simulator import (
    calibrate_alphas,
    calibrate_layer_alphas,
    run_training,
)
from compressiq.worker import get_layer_offsets, layer_bits as compute_layer_bits


def _build_profiles(spec: str, n: int, seed: int):
    presets = {
        "2tier": None,
        "3tier": [(10.0, 5.0, 0.5), (5.0, 20.0, 0.25), (1.0, 50.0, 0.25)],
        "4tier": [(10.0, 5.0, 0.375), (5.0, 15.0, 0.25),
                  (2.5, 30.0, 0.25), (1.0, 50.0, 0.125)],
    }
    if spec == "2tier":
        return make_heterogeneous_profiles(n, seed=seed)
    if spec in presets:
        return make_tiered_profiles(n, presets[spec], seed=seed)
    tiers = [tuple(float(v) for v in part.split(":")) for part in spec.split(",")]
    return make_tiered_profiles(n, tiers, seed=seed)


def _make_resolve_fn(variant: str, profiles, layer_bits_arr, grad_bits, epsilon, r_min):
    """Return a resolve_fn compatible with simulator.run_training."""
    if variant == "naive":
        def fn(alphas, **_):
            return solve_compressiq(profiles, grad_bits, alphas, epsilon, r_min=r_min).ratios
        return fn
    if variant == "ef":
        def fn(alphas, kappa=None, **_):
            return solve_compressiq(
                profiles, grad_bits, alphas, epsilon, r_min=r_min,
                ef_factors=kappa,
            ).ratios
        return fn
    if variant == "per_layer":
        def fn(alphas_matrix, **_):
            return solve_compressiq_per_layer(
                profiles, layer_bits_arr, alphas_matrix, epsilon, r_min=r_min,
            ).ratios
        return fn
    if variant == "per_layer_ef":
        def fn(alphas_matrix, kappa=None, **_):
            return solve_compressiq_per_layer(
                profiles, layer_bits_arr, alphas_matrix, epsilon, r_min=r_min,
                ef_factors=kappa,
            ).ratios
        return fn
    raise ValueError(variant)


def _initial_ratios(variant, profiles, layer_bits_arr, grad_bits, alphas_scalar,
                    alphas_matrix, epsilon, r_min):
    """Solve once for starting ratios, matching the variant's problem shape."""
    if variant in ("naive", "ef"):
        return solve_compressiq(profiles, grad_bits, alphas_scalar, epsilon, r_min=r_min).ratios
    return solve_compressiq_per_layer(
        profiles, layer_bits_arr, alphas_matrix, epsilon, r_min=r_min,
    ).ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--r-min", type=float, default=0.01)
    ap.add_argument("--recalibrate-every", type=int, default=25)
    ap.add_argument("--epsilon-frac", type=float, default=0.15,
                    help="epsilon as fraction of initial sum of alphas")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiers", type=str, default="3tier")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    profiles = _build_profiles(args.tiers, args.n_workers, args.seed)
    print("Network profiles (Gbps / lat us):")
    for i, p in enumerate(profiles):
        print(f"  W{i}: {p.bandwidth_bps/1e9:5.2f} Gbps   {p.latency_s*1e6:6.2f} us")

    # One cluster build to harvest grad_bits + layer_bits + initial alphas.
    workers, test_loader, grad_bits = build_cluster(
        args.n_workers, profiles, batch_size=args.batch_size,
        device=args.device, seed=args.seed, use_error_feedback=True,
    )
    layer_bits_arr = np.array(compute_layer_bits(workers[0].model), dtype=float)
    print(f"Gradient size: {grad_bits} bits ({grad_bits/8/1024:.1f} KB)")
    print(f"Layers ({len(layer_bits_arr)}): {layer_bits_arr.astype(int).tolist()} bits")

    print("Calibrating initial alphas (scalar + per-layer) ...")
    alphas_scalar = calibrate_alphas(workers)
    alphas_matrix = calibrate_layer_alphas(workers)
    print("scalar alphas:", np.round(alphas_scalar, 3))
    print("matrix alphas shape:", alphas_matrix.shape)

    epsilon = args.epsilon_frac * float(alphas_scalar.sum())
    print(f"epsilon = {args.epsilon_frac:.3f} * sum(alpha) = {epsilon:.3f}")

    variants = ["naive", "ef", "per_layer", "per_layer_ef"]
    full_rows = []
    summary_rows = []

    for variant in variants:
        print(f"\n=== variant={variant}  eps={epsilon:.3f} ===")
        # Fresh cluster each run so initial weights are identical.
        workers, test_loader, grad_bits = build_cluster(
            args.n_workers, profiles, batch_size=args.batch_size,
            device=args.device, seed=args.seed, use_error_feedback=True,
        )

        resolve_fn = _make_resolve_fn(
            variant, profiles, layer_bits_arr, grad_bits, epsilon, args.r_min,
        )
        init_ratios = _initial_ratios(
            variant, profiles, layer_bits_arr, grad_bits, alphas_scalar,
            alphas_matrix, epsilon, args.r_min,
        )

        logs = run_training(
            workers, init_ratios, grad_bits, test_loader,
            num_rounds=args.rounds, lr=args.lr, eval_every=args.eval_every,
            device=args.device, verbose=True, epsilon=epsilon,
            recalibrate_every=args.recalibrate_every, resolve_fn=resolve_fn,
            layer_bits_vec=layer_bits_arr,
        )

        for L in logs:
            full_rows.append({
                "variant": variant,
                "round": L.round_idx,
                "sim_time_s": L.sim_time_s,
                "round_time_s": L.round_time_s,
                "bottleneck_worker": L.bottleneck_worker,
                "test_accuracy": L.test_accuracy,
                "test_loss": L.test_loss,
                "theoretical_error": L.theoretical_error,
                "empirical_error": L.empirical_error,
                "kappa_mean": float(L.kappa_ema.mean()) if L.kappa_ema.size else 1.0,
                "epsilon": L.epsilon,
                "recalibrated": L.recalibrated,
            })
        final = logs[-1]
        summary_rows.append({
            "variant": variant,
            "final_accuracy": final.test_accuracy,
            "final_loss": final.test_loss,
            "total_sim_time_s": final.sim_time_s,
            "final_theoretical": final.theoretical_error,
            "final_empirical": final.empirical_error,
            "final_kappa_mean": float(final.kappa_ema.mean()) if final.kappa_ema.size else 1.0,
            "final_ratios_shape": str(final.ratios.shape),
        })

    pd.DataFrame(full_rows).to_csv(os.path.join(args.out, "refine_rounds.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out, "refine_summary.csv"), index=False)
    print("\nWrote", os.path.join(args.out, "refine_rounds.csv"))
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
