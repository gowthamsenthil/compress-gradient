"""Adaptive recalibration ablation + bound-validation diagnostics.

Compares the static CompressIQ solution against periodic recalibration
(every K rounds re-measure alpha_i and re-solve the convex program).

Produces a 4-panel figure:
  Panel 1: cumulative simulated time vs round
  Panel 2: test accuracy vs round
  Panel 3: per-round theoretical and empirical error vs the budget epsilon
           (the "growing slack" plot -- the headline result)
  Panel 4: per-worker alpha drift over training (||g_i(t)||^2)

Outputs:
  results/adaptive_rounds.csv  per-round logs across all schemes
  results/adaptive.png         4-panel figure
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compressiq.cluster import build_cluster, make_heterogeneous_profiles
from compressiq.optimizer import solve_compressiq
from compressiq.simulator import calibrate_alphas, run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--epsilon-frac", type=float, default=0.3,
                    help="epsilon = frac * sum(alpha_i) at calibration time")
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 50, 25, 10],
                    help="Recalibration intervals; 0 = static (no recalibration)")
    ap.add_argument("--r-min", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--no-ef", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    profiles = make_heterogeneous_profiles(args.n_workers, seed=args.seed)
    print("Network profiles (Gbps / latency us):")
    for i, p in enumerate(profiles):
        print(f"  W{i}: {p.bandwidth_bps/1e9:5.2f} Gbps   {p.latency_s*1e6:6.2f} us")

    # One initial calibration to fix epsilon.
    workers, test_loader, grad_bits = build_cluster(
        args.n_workers, profiles, batch_size=args.batch_size,
        device=args.device, seed=args.seed, use_error_feedback=not args.no_ef,
    )
    alphas0 = calibrate_alphas(workers)
    epsilon = args.epsilon_frac * float(np.sum(alphas0))
    print(f"Initial alphas: {np.round(alphas0,3)}")
    print(f"epsilon = {epsilon:.4f}  (= {args.epsilon_frac} * sum(alpha))")

    rows = []
    for K in args.ks:
        # Fresh cluster each run (same init weights via seed).
        workers, test_loader, grad_bits = build_cluster(
            args.n_workers, profiles, batch_size=args.batch_size,
            device=args.device, seed=args.seed, use_error_feedback=not args.no_ef,
        )

        # Initial solution from the calibration alphas
        ratios0 = solve_compressiq(profiles, grad_bits, alphas0, epsilon, r_min=args.r_min).ratios

        if K == 0:
            scheme = "static"
            recal, resolve_fn = None, None
        else:
            scheme = f"adaptive-K{K}"
            recal = K
            # closure: take fresh alphas, return fresh ratios
            def make_resolver(eps):
                def f(alphas_new):
                    return solve_compressiq(profiles, grad_bits, alphas_new, eps, r_min=args.r_min).ratios
                return f
            resolve_fn = make_resolver(epsilon)

        print(f"\n=== scheme={scheme}  ratios0={np.round(ratios0,3)} ===")
        logs = run_training(
            workers, ratios0, grad_bits, test_loader,
            num_rounds=args.rounds, lr=args.lr,
            eval_every=args.eval_every, device=args.device, verbose=False,
            epsilon=epsilon, recalibrate_every=recal, resolve_fn=resolve_fn,
        )

        for L in logs:
            rows.append({
                "scheme": scheme, "K": K, "round": L.round_idx,
                "sim_time_s": L.sim_time_s, "round_time_s": L.round_time_s,
                "test_accuracy": L.test_accuracy, "test_loss": L.test_loss,
                "theoretical_error": L.theoretical_error,
                "empirical_error": L.empirical_error,
                "epsilon": L.epsilon,
                "recalibrated": L.recalibrated,
                **{f"alpha_w{i}": float(L.alpha_now[i]) for i in range(len(L.alpha_now))},
                **{f"r_w{i}": float(L.ratios[i]) for i in range(len(L.ratios))},
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "adaptive_rounds.csv"), index=False)
    print(f"\nWrote {os.path.join(args.out, 'adaptive_rounds.csv')}  ({len(df)} rows)")

    _make_plots(df, args, epsilon)


def _make_plots(df: pd.DataFrame, args, epsilon: float):
    import matplotlib.pyplot as plt

    schemes = list(df["scheme"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Panel 1: cumulative simulated time
    for s in schemes:
        sub = df[df["scheme"] == s].sort_values("round")
        ax1.plot(sub["round"], sub["sim_time_s"], label=s)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Cumulative simulated time (s)")
    ax1.set_title("Communication time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: test accuracy
    for s in schemes:
        sub = df[df["scheme"] == s].sort_values("round")
        ax2.plot(sub["round"], sub["test_accuracy"], label=s)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Test accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: bound validation -- the headline plot
    for s in schemes:
        sub = df[df["scheme"] == s].sort_values("round")
        line, = ax3.plot(sub["round"], sub["theoretical_error"],
                         label=f"{s} theoretical", linestyle="-")
        ax3.plot(sub["round"], sub["empirical_error"],
                 label=f"{s} empirical", linestyle="--", color=line.get_color(), alpha=0.7)
    ax3.axhline(epsilon, color="k", linestyle=":", linewidth=1.5, label=f"epsilon={epsilon:.3f}")
    ax3.set_xlabel("Round")
    ax3.set_ylabel(r"$\Sigma_i \alpha_i (1 - r_i)^2$")
    ax3.set_title("Bound validation: theoretical (solid) vs empirical (dashed)")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.3)

    # Panel 4: alpha drift (use the static run; alpha_i is a property of training,
    # not of the scheme, so any one run suffices)
    static = df[df["scheme"] == "static"].sort_values("round")
    alpha_cols = [c for c in df.columns if c.startswith("alpha_w")]
    for c in alpha_cols:
        ax4.plot(static["round"], static[c], label=c.replace("alpha_w", "W"), alpha=0.8)
    ax4.set_xlabel("Round")
    ax4.set_ylabel(r"$\alpha_i^{\mathrm{true}}(t) = \|g_i(t)\|^2$")
    ax4.set_title(r"Per-worker $\alpha_i$ drift during training")
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale("log")

    fig.suptitle(
        f"Adaptive recalibration ablation  "
        f"(N={args.n_workers}, rounds={args.rounds}, "
        f"eps={epsilon:.3f}, EF={'off' if args.no_ef else 'on'})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "adaptive.png"), dpi=140)
    plt.close(fig)
    print(f"Wrote {os.path.join(args.out, 'adaptive.png')}")


if __name__ == "__main__":
    main()
