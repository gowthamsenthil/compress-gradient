"""Pareto frontier: communication time vs. accuracy across baselines.

Sweeps the accuracy budget epsilon and runs four schemes:
  - none     : r_i = 1 (no compression)
  - uniform  : single r* across all workers, satisfies budget
  - greedy   : per-worker r_i, no coordination (slower workers get more budget)
  - compressiq: convex-optimal per-worker r_i (the proposed method)

Outputs:
  results/pareto.csv   : full per-round logs across all (scheme, epsilon) runs
  results/summary.csv  : final-round summary per (scheme, epsilon)
  results/pareto.png   : Pareto frontier
  results/ratios.png   : per-worker compression-ratio bar chart at one epsilon
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Allow running from project root without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compressiq.cluster import (
    build_cluster,
    make_heterogeneous_profiles,
    make_tiered_profiles,
)
from compressiq.cost_model import per_worker_time
from compressiq.optimizer import (
    baseline_greedy,
    baseline_none,
    baseline_uniform,
    solve_compressiq,
)
from compressiq.simulator import calibrate_alphas, run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=15)
    ap.add_argument("--r-min", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--no-ef", action="store_true", help="Disable error feedback")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument(
        "--tiers",
        type=str,
        default="2tier",
        help=(
            "Cluster heterogeneity spec. Preset names: '2tier' (default, 10+1 Gbps), "
            "'3tier' (10+5+1 Gbps), '4tier' (10+5+2.5+1 Gbps). Or provide a custom "
            "spec as 'bw:lat:frac,bw:lat:frac,...' e.g. '10:5:0.5,5:20:0.25,1:50:0.25'"
        ),
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Heterogeneous cluster
    profiles = _build_profiles(args.tiers, args.n_workers, args.seed)
    print("Network profiles (Gbps / latency us):")
    for i, p in enumerate(profiles):
        print(f"  W{i}: {p.bandwidth_bps/1e9:5.2f} Gbps   {p.latency_s*1e6:6.2f} us")

    # Build cluster once to estimate alpha_i; rebuild for each run to reset weights
    workers, test_loader, grad_bits = build_cluster(
        args.n_workers, profiles, batch_size=args.batch_size,
        device=args.device, seed=args.seed, use_error_feedback=not args.no_ef,
    )
    print(f"Gradient size: {grad_bits} bits ({grad_bits/8/1024:.1f} KB)")

    print("Calibrating alpha_i = ||g_i||^2 ...")
    alphas = calibrate_alphas(workers)
    print("alphas:", np.round(alphas, 3))

    # Sensible epsilon range based on the calibration
    alpha_sum = float(np.sum(alphas))
    epsilons = np.geomspace(0.01 * alpha_sum, 0.9 * alpha_sum, num=6)
    print("epsilon sweep:", np.round(epsilons, 3))

    schemes = ["none", "uniform", "greedy", "compressiq"]

    summary_rows = []
    full_rows = []

    for eps in epsilons:
        for scheme in schemes:
            if scheme == "none":
                ratios = baseline_none(args.n_workers)
            elif scheme == "uniform":
                ratios = baseline_uniform(args.n_workers, alphas, eps, r_min=args.r_min)
            elif scheme == "greedy":
                ratios = baseline_greedy(profiles, alphas, eps, r_min=args.r_min)
            elif scheme == "compressiq":
                res = solve_compressiq(profiles, grad_bits, alphas, eps, r_min=args.r_min)
                ratios = res.ratios
            else:
                raise ValueError(scheme)

            # Fresh cluster every run (same init weights via seed).
            workers, test_loader, grad_bits = build_cluster(
                args.n_workers, profiles, batch_size=args.batch_size,
                device=args.device, seed=args.seed, use_error_feedback=not args.no_ef,
            )

            print(f"\n=== scheme={scheme:11s}  eps={eps:8.3f}  ratios={np.round(ratios,3)} ===")
            logs = run_training(
                workers, ratios, grad_bits, test_loader,
                num_rounds=args.rounds, lr=args.lr,
                eval_every=args.eval_every, device=args.device, verbose=True,
            )

            for L in logs:
                full_rows.append({
                    "scheme": scheme, "epsilon": eps, "round": L.round_idx,
                    "sim_time_s": L.sim_time_s, "round_time_s": L.round_time_s,
                    "bottleneck_worker": L.bottleneck_worker,
                    "test_accuracy": L.test_accuracy, "test_loss": L.test_loss,
                })
            final = logs[-1]
            times = per_worker_time(ratios, grad_bits, profiles)
            summary_rows.append({
                "scheme": scheme, "epsilon": eps,
                "final_accuracy": final.test_accuracy,
                "final_loss": final.test_loss,
                "total_sim_time_s": final.sim_time_s,
                "round_time_s_max": float(times.max()),
                "round_time_s_mean": float(times.mean()),
                "ratios": ",".join(f"{r:.4f}" for r in ratios),
            })

    pd.DataFrame(full_rows).to_csv(os.path.join(args.out, "pareto_rounds.csv"), index=False)
    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("\nWrote", os.path.join(args.out, "summary.csv"))
    print(df.to_string(index=False))

    # Plots
    try:
        _make_plots(df, profiles, alphas, args)
    except Exception as e:
        print(f"[warn] plotting failed: {e}")


def _build_profiles(spec: str, n: int, seed: int):
    """Parse --tiers into a profile list."""
    presets = {
        "2tier": [(10.0, 5.0, 0.75), (1.0, 50.0, 0.25)],
        "3tier": [(10.0, 5.0, 0.5), (5.0, 20.0, 0.25), (1.0, 50.0, 0.25)],
        "4tier": [(10.0, 5.0, 0.375), (5.0, 15.0, 0.25),
                  (2.5, 30.0, 0.25), (1.0, 50.0, 0.125)],
    }
    if spec in presets:
        tiers = presets[spec]
    else:
        # Custom spec: 'bw:lat:frac,bw:lat:frac,...'
        tiers = []
        for part in spec.split(","):
            bw, lat, frac = part.split(":")
            tiers.append((float(bw), float(lat), float(frac)))
    # Fall back to the old 2-tier function when that preset is requested so we
    # keep the exact same random placement as earlier runs (reproducibility).
    if spec == "2tier":
        return make_heterogeneous_profiles(n, seed=seed)
    return make_tiered_profiles(n, tiers, seed=seed)


def _make_plots(df: pd.DataFrame, profiles, alphas, args):
    import matplotlib.pyplot as plt

    # 1. Pareto: communication time vs final accuracy
    fig, ax = plt.subplots(figsize=(7, 5))
    for scheme, sub in df.groupby("scheme"):
        sub = sub.sort_values("epsilon")
        ax.plot(sub["total_sim_time_s"], sub["final_accuracy"],
                marker="o", label=scheme)
    ax.set_xlabel("Total simulated communication time (s)")
    ax.set_ylabel("Final test accuracy")
    ax.set_title(f"Pareto frontier (N={args.n_workers}, rounds={args.rounds})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "pareto.png"), dpi=140)
    plt.close(fig)

    # 2. Per-worker compression ratios at the median epsilon, for compressiq vs uniform
    eps_vals = sorted(df["epsilon"].unique())
    eps_mid = eps_vals[len(eps_vals) // 2]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    x = np.arange(args.n_workers)
    for off, scheme in zip([-width / 2, width / 2], ["uniform", "compressiq"]):
        row = df[(df["scheme"] == scheme) & (df["epsilon"] == eps_mid)].iloc[0]
        ratios = np.array([float(v) for v in row["ratios"].split(",")])
        ax.bar(x + off, ratios, width, label=scheme)
    bws = np.array([p.bandwidth_bps / 1e9 for p in profiles])
    ax2 = ax.twinx()
    ax2.plot(x, bws, "k--o", alpha=0.5, label="bandwidth (Gbps)")
    ax.set_xlabel("Worker")
    ax.set_ylabel("Compression ratio r_i")
    ax2.set_ylabel("Bandwidth (Gbps)")
    ax.set_xticks(x)
    ax.set_title(f"Per-worker r_i  (epsilon={eps_mid:.3f})")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "ratios.png"), dpi=140)
    plt.close(fig)

    print("Wrote plots: pareto.png, ratios.png")


if __name__ == "__main__":
    main()
