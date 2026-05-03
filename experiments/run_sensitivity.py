"""Sensitivity to cluster size N. Reports the ratio of round_time(compressiq) /
round_time(uniform) at a fixed epsilon, for several N. Demonstrates the
2(N-1)/N factor and the bottleneck-mitigation effect."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compressiq.cluster import build_cluster, make_heterogeneous_profiles
from compressiq.cost_model import per_worker_time
from compressiq.optimizer import baseline_uniform, solve_compressiq
from compressiq.simulator import calibrate_alphas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ns", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--epsilon-frac", type=float, default=0.3,
                    help="epsilon = frac * sum(alpha_i)")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for N in args.Ns:
        profiles = make_heterogeneous_profiles(N, seed=args.seed)
        workers, _, grad_bits = build_cluster(N, profiles, seed=args.seed)
        alphas = calibrate_alphas(workers)
        eps = args.epsilon_frac * float(np.sum(alphas))

        r_uni = baseline_uniform(N, alphas, eps)
        r_ciq = solve_compressiq(profiles, grad_bits, alphas, eps).ratios

        t_uni = per_worker_time(r_uni, grad_bits, profiles).max()
        t_ciq = per_worker_time(r_ciq, grad_bits, profiles).max()
        speedup = t_uni / t_ciq if t_ciq > 0 else float("inf")

        rows.append({
            "N": N, "epsilon": eps, "grad_bits": grad_bits,
            "round_time_uniform_s": t_uni,
            "round_time_compressiq_s": t_ciq,
            "speedup": speedup,
            "ring_factor": 2 * (N - 1) / N,
        })
        print(f"N={N:3d}  uniform={t_uni*1e3:7.2f}ms  compressiq={t_ciq*1e3:7.2f}ms  "
              f"speedup={speedup:5.2f}x  2(N-1)/N={2*(N-1)/N:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "sensitivity_N.csv"), index=False)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(df["N"], df["round_time_uniform_s"] * 1e3, "-o", label="uniform")
        ax.plot(df["N"], df["round_time_compressiq_s"] * 1e3, "-o", label="compressiq")
        ax.set_xlabel("Cluster size N")
        ax.set_ylabel("Per-round bottleneck time (ms)")
        ax.set_title(f"Round time vs N  (epsilon = {args.epsilon_frac:.2f} * sum alpha)")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "sensitivity_N.png"), dpi=140)
        plt.close(fig)
        print("Wrote sensitivity_N.png")
    except Exception as e:
        print(f"[warn] plotting failed: {e}")


if __name__ == "__main__":
    main()
