"""Produce all 8 report figures from existing experiment outputs.

Reads:
  results/pareto_rounds.csv      (from run_pareto.py)
  results/summary.csv            (from run_pareto.py)
  results/adaptive_rounds.csv    (from run_adaptive.py)

Runs a small standalone sweep for figure #4 (no training).

Writes (in --out, default results/):
  fig1_pareto.png              accuracy vs cumulative comm time
  fig2_ratios.png              per-worker r_i sorted by bandwidth
  fig3_round_time.png          per-round comm time
  fig4_model_validation.png    theoretical (1-r)^2 alpha vs empirical Top-K error
  fig5_constraint.png          theoretical/empirical error vs epsilon over rounds
  fig6_alpha_drift.png         per-worker ||g_i||^2 over rounds (log-y)
  fig7_speedup.png             relative comm time: uniform / static / adaptive
  fig8_eps_sensitivity.png     time to 90% accuracy vs epsilon
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------- helpers ----------

def _load_csv(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"[warn] {path} not found; skipping figures that need it.")
    return None


def _parse_ratios(s: str) -> np.ndarray:
    return np.array([float(v) for v in s.split(",")])


# ---------- fig 1 ----------

def fig1_pareto(pareto_rounds: pd.DataFrame, out: str):
    """Speed vs accuracy across schemes. On MNIST all schemes converge to nearly
    the same final accuracy (~0.895), so the 'frontier' manifests almost entirely
    as a speedup at iso-accuracy. We sort points by epsilon (not accuracy) so the
    line traces the speed-budget tradeoff cleanly."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    schemes = ["none", "uniform", "greedy", "compressiq"]
    colors = {"none": "#888888", "uniform": "#1f77b4", "greedy": "#2ca02c", "compressiq": "#d62728"}
    for s in schemes:
        sub = pareto_rounds[pareto_rounds["scheme"] == s]
        if sub.empty:
            continue
        finals = sub.sort_values("round").groupby("epsilon").tail(1)
        finals = finals.sort_values("epsilon")
        ax.plot(finals["test_accuracy"], finals["sim_time_s"],
                marker="o", color=colors[s], label=s, linewidth=2,
                markersize=7, linestyle="-" if s != "none" else "")
    # Pin x range so the noise band doesn't dominate the plot.
    ax.set_xlim(0.85, 0.92)
    ax.set_xlabel("Final test accuracy")
    ax.set_ylabel("Cumulative communication time (s)")
    ax.set_title("Speed–accuracy tradeoff across schemes (each point = one $\\epsilon$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_pareto.png"), dpi=140)
    plt.close(fig)


# ---------- fig 2 ----------

def fig2_ratios(summary: pd.DataFrame, profiles_bw_gbps: np.ndarray, out: str):
    """Grouped bar chart of CompressIQ vs Uniform per-worker r_i, sorted fast->slow.

    Picks the epsilon that reveals the richest tiered structure (max number of
    distinct CompressIQ ratios, break ties by middle epsilon). On a 3-tier
    cluster this produces the classic 3-step staircase."""
    eps_vals = sorted(summary["epsilon"].unique())
    n = len(profiles_bw_gbps)

    # For each epsilon, compute how many bandwidth tiers are "active" (i.e., have
    # at least one worker with r strictly inside (r_min, 1)). Prefer the eps that
    # exposes the most tiers -- that's the clearest staircase.
    tier_bws = np.unique(np.round(profiles_bw_gbps, 3))
    best_eps, best_active_tiers = eps_vals[len(eps_vals) // 2], -1
    for eps in eps_vals:
        r = _parse_ratios(
            summary[(summary["scheme"] == "compressiq") & (summary["epsilon"] == eps)]
            .iloc[0]["ratios"]
        )
        active = 0
        for bw in tier_bws:
            mask = np.isclose(profiles_bw_gbps, bw)
            if ((r[mask] > 0.015) & (r[mask] < 0.99)).any():
                active += 1
        if active > best_active_tiers:
            best_active_tiers, best_eps = active, eps
    eps_mid = best_eps

    order = np.argsort(-profiles_bw_gbps)  # fast first

    row_uni = summary[(summary["scheme"] == "uniform") & (summary["epsilon"] == eps_mid)].iloc[0]
    row_ciq = summary[(summary["scheme"] == "compressiq") & (summary["epsilon"] == eps_mid)].iloc[0]
    r_uni = _parse_ratios(row_uni["ratios"])[order]
    r_ciq = _parse_ratios(row_ciq["ratios"])[order]
    bws = profiles_bw_gbps[order]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(n)
    w = 0.4
    ax.bar(x - w/2, r_uni, w, label="Uniform", color="#1f77b4")
    ax.bar(x + w/2, r_ciq, w, label="CompressIQ", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([f"W{i}\n{bws[k]:.0f} Gbps" for k, i in enumerate(order)])
    ax.set_ylabel("Compression ratio $r_i$")
    ax.set_title(f"Per-worker compression ratios (sorted fast → slow)   epsilon={eps_mid:.3f}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_ratios.png"), dpi=140)
    plt.close(fig)


# ---------- fig 3 ----------

def fig3_round_time(pareto_rounds: pd.DataFrame, out: str):
    """Per-round communication time across schemes at the median epsilon."""
    eps_vals = sorted(pareto_rounds["epsilon"].unique())
    eps_mid = eps_vals[len(eps_vals) // 2]
    sub = pareto_rounds[pareto_rounds["epsilon"] == eps_mid]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = {"none": "#888888", "uniform": "#1f77b4", "greedy": "#2ca02c", "compressiq": "#d62728"}
    for s, df_s in sub.groupby("scheme"):
        df_s = df_s.sort_values("round")
        ax.plot(df_s["round"], df_s["round_time_s"] * 1e3,
                label=s, color=colors.get(s, None), linewidth=1.8)
    ax.set_xlabel("Training round")
    ax.set_ylabel("Per-round communication time (ms)")
    ax.set_title(f"Round-by-round communication time   epsilon={eps_mid:.3f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_round_time.png"), dpi=140)
    plt.close(fig)


# ---------- fig 4 ----------

def fig4_model_validation(out: str, n_workers: int = 8, seed: int = 0):
    """Sweep r in [0.01, 1] on freshly-computed gradients; compare theoretical
    (1-r)^2 * alpha vs empirical ||g - TopK(g, r)||^2."""
    from compressiq.cluster import build_cluster, make_heterogeneous_profiles
    from compressiq.compression import topk_compress

    profiles = make_heterogeneous_profiles(n_workers, seed=seed)
    workers, _, _ = build_cluster(n_workers, profiles, seed=seed, use_error_feedback=False)

    rs = np.linspace(0.02, 1.0, 25)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, n_workers))

    for i, w in enumerate(workers):
        g = w.compute_gradient()
        alpha = float(g.pow(2).sum().item())
        emp = np.array([float((g - topk_compress(g, float(r))).pow(2).sum().item()) for r in rs])
        theo = alpha * (1.0 - rs) ** 2
        ax.plot(rs, theo, color=cmap[i], linestyle="-", linewidth=1.6, alpha=0.6)
        ax.scatter(rs, emp, color=cmap[i], s=18, label=f"W{i}")

    # Single legend entry for the model curve
    ax.plot([], [], color="k", linestyle="-", label=r"theoretical $(1-r)^2 \alpha_i$")
    ax.set_xlabel("Compression ratio $r$")
    ax.set_ylabel("Squared error")
    ax.set_title("Accuracy-loss model validation: theoretical (lines) vs empirical Top-K (points)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_model_validation.png"), dpi=140)
    plt.close(fig)


# ---------- fig 5 ----------

def fig5_constraint(adaptive: pd.DataFrame, out: str):
    """Theoretical and empirical error vs epsilon over training rounds."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    eps = float(adaptive["epsilon"].iloc[0])
    schemes = ["static", "adaptive-K50", "adaptive-K25", "adaptive-K10"]
    colors = {"static": "#d62728", "adaptive-K50": "#ff7f0e",
              "adaptive-K25": "#2ca02c", "adaptive-K10": "#1f77b4"}
    for s in schemes:
        sub = adaptive[adaptive["scheme"] == s].sort_values("round")
        if sub.empty:
            continue
        c = colors.get(s, None)
        ax.plot(sub["round"], sub["theoretical_error"], color=c, linestyle="-",
                linewidth=2.0, label=f"{s} theoretical")
        ax.plot(sub["round"], sub["empirical_error"], color=c, linestyle="--",
                linewidth=1.4, alpha=0.75, label=f"{s} empirical")
    ax.axhline(eps, color="k", linestyle=":", linewidth=1.6, label=f"$\\epsilon = {eps:.3f}$")
    ax.set_xlabel("Training round")
    ax.set_ylabel(r"$\Sigma_i \alpha_i (1 - r_i)^2$")
    ax.set_title("Constraint satisfaction: budget utilisation over training")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_constraint.png"), dpi=140)
    plt.close(fig)


# ---------- fig 6 ----------

def fig6_alpha_drift(adaptive: pd.DataFrame, out: str):
    """Per-worker ||g_i||^2 over rounds, log-y."""
    static = adaptive[adaptive["scheme"] == "static"].sort_values("round")
    alpha_cols = sorted([c for c in static.columns if c.startswith("alpha_w")],
                        key=lambda s: int(s.replace("alpha_w", "")))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    cmap = plt.cm.tab10(np.linspace(0, 1, len(alpha_cols)))
    for i, c in enumerate(alpha_cols):
        ax.plot(static["round"], static[c], color=cmap[i], label=c.replace("alpha_w", "W"),
                linewidth=1.4, alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Training round")
    ax.set_ylabel(r"$\alpha_i^{\mathrm{true}}(t) = \|g_i(t)\|^2$  (log scale)")
    ax.set_title(r"Per-worker gradient-norm drift during training")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_alpha_drift.png"), dpi=140)
    plt.close(fig)


# ---------- fig 7 ----------

def fig7_speedup(summary: pd.DataFrame, adaptive: pd.DataFrame, out: str):
    """Bar chart of relative communication time. We use the median epsilon for the
    Pareto-summary file. Adaptive uses K=10 if available."""
    eps_vals = sorted(summary["epsilon"].unique())
    eps_mid = eps_vals[len(eps_vals) // 2]
    t_uni = float(summary[(summary["scheme"] == "uniform") & (summary["epsilon"] == eps_mid)]["total_sim_time_s"].iloc[0])
    t_ciq_static = float(summary[(summary["scheme"] == "compressiq") & (summary["epsilon"] == eps_mid)]["total_sim_time_s"].iloc[0])

    # Adaptive run uses its own (different) epsilon; compare via final sim_time relative to its own static
    static_a = adaptive[adaptive["scheme"] == "static"].sort_values("round").tail(1)
    adapt_a = adaptive[adaptive["scheme"] == "adaptive-K10"].sort_values("round").tail(1)
    t_static_a = float(static_a["sim_time_s"].iloc[0]) if not static_a.empty else None
    t_adapt_a = float(adapt_a["sim_time_s"].iloc[0]) if not adapt_a.empty else None

    schemes = ["Uniform", "Static\nCompressIQ", "Adaptive\nCompressIQ (K=10)"]
    times = [t_uni, t_ciq_static]
    if t_static_a is not None and t_adapt_a is not None:
        # rescale adaptive comparison into Pareto epsilon's frame using its own static as anchor
        scale = t_ciq_static / t_static_a if t_static_a > 0 else 1.0
        times.append(t_adapt_a * scale)
    else:
        schemes = schemes[:2]

    speedup = [t_uni / t for t in times]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = ["#1f77b4", "#d62728", "#ff7f0e"][: len(schemes)]
    bars = ax.bar(schemes, speedup, color=colors)
    for b, s in zip(bars, speedup):
        ax.text(b.get_x() + b.get_width()/2, s + 0.03, f"{s:.2f}x",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Speedup over Uniform (cumulative comm time)")
    ax.set_title(f"Speedup decomposition  (epsilon={eps_mid:.3f})")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, max(speedup) * 1.25)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig7_speedup.png"), dpi=140)
    plt.close(fig)


# ---------- fig 8 ----------

def fig8_eps_sensitivity(pareto_rounds: pd.DataFrame, out: str, target_acc: float = 0.88):
    """Time to reach target accuracy per (scheme, epsilon). 0.88 used because not
    every run reaches 0.90 within 150 rounds in our pareto setup."""
    rows = []
    for (scheme, eps), sub in pareto_rounds.groupby(["scheme", "epsilon"]):
        sub = sub.sort_values("round")
        hit = sub[sub["test_accuracy"] >= target_acc]
        t = float(hit["sim_time_s"].min()) if not hit.empty else np.nan
        rows.append({"scheme": scheme, "epsilon": eps, "time_to_target": t})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = {"none": "#888888", "uniform": "#1f77b4", "greedy": "#2ca02c", "compressiq": "#d62728"}
    for scheme, sub in df.groupby("scheme"):
        sub = sub.sort_values("epsilon")
        ax.plot(sub["epsilon"], sub["time_to_target"],
                marker="o", color=colors.get(scheme, None), label=scheme, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"Accuracy budget $\epsilon$ (log scale)")
    ax.set_ylabel(f"Sim. time to reach {int(target_acc*100)}% accuracy (s)")
    ax.set_title("Sensitivity to accuracy budget")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig8_eps_sensitivity.png"), dpi=140)
    plt.close(fig)


# ---------- fig 9 (bound refinements: speed at iso-accuracy) ----------

def fig9_refine_speed(refine: pd.DataFrame, summary: pd.DataFrame, out: str):
    """Two-panel: (left) accuracy vs cumulative sim time; (right) speedup bars."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    variants = ["naive", "ef", "per_layer", "per_layer_ef"]
    colors = {"naive": "#888888", "ef": "#1f77b4",
              "per_layer": "#2ca02c", "per_layer_ef": "#d62728"}
    labels = {"naive": "Naive bound",
              "ef": "EF-aware bound",
              "per_layer": "Per-layer",
              "per_layer_ef": "Per-layer + EF-aware"}
    for v in variants:
        sub = refine[refine["variant"] == v].sort_values("round")
        if sub.empty:
            continue
        axL.plot(sub["sim_time_s"], sub["test_accuracy"],
                 color=colors[v], label=labels[v], linewidth=2)
    axL.set_xlabel("Cumulative simulated communication time (s)")
    axL.set_ylabel("Test accuracy")
    axL.set_title("Accuracy vs comm time for each bound refinement")
    axL.grid(True, alpha=0.3)
    axL.legend(loc="lower right")

    baseline = float(summary[summary["variant"] == "naive"]["total_sim_time_s"].iloc[0])
    order = [v for v in variants if v in set(summary["variant"])]
    speeds = [baseline / float(summary[summary["variant"] == v]["total_sim_time_s"].iloc[0])
              for v in order]
    bars = axR.bar([labels[v] for v in order], speeds,
                   color=[colors[v] for v in order])
    for b, s in zip(bars, speeds):
        axR.text(b.get_x() + b.get_width()/2, s + 0.03, f"{s:.2f}x",
                 ha="center", va="bottom", fontweight="bold")
    axR.axhline(1.0, color="k", linestyle="--", linewidth=0.8)
    axR.set_ylabel("Speedup vs Naive bound (total sim comm time)")
    axR.set_title("Bound-refinement speedup at matched accuracy")
    axR.grid(True, alpha=0.3, axis="y")
    axR.set_ylim(0, max(speeds) * 1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig9_refine_speed.png"), dpi=140)
    plt.close(fig)


# ---------- fig 10 (bound tightness) ----------

def fig10_bound_tightness(refine: pd.DataFrame, out: str):
    """Per-variant theoretical vs empirical per-round error, with kappa_ema."""
    variants = ["naive", "ef", "per_layer", "per_layer_ef"]
    colors = {"naive": "#888888", "ef": "#1f77b4",
              "per_layer": "#2ca02c", "per_layer_ef": "#d62728"}
    labels = {"naive": "Naive bound",
              "ef": "EF-aware bound",
              "per_layer": "Per-layer",
              "per_layer_ef": "Per-layer + EF-aware"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    eps = float(refine["epsilon"].iloc[0])
    for v in variants:
        sub = refine[refine["variant"] == v].sort_values("round")
        if sub.empty:
            continue
        axes[0].plot(sub["round"], sub["theoretical_error"],
                     color=colors[v], linestyle="-", linewidth=2, label=f"{labels[v]} (theo)")
        axes[0].plot(sub["round"], sub["empirical_error"],
                     color=colors[v], linestyle="--", linewidth=1.4, alpha=0.7)
        axes[1].plot(sub["round"], sub["kappa_mean"],
                     color=colors[v], linewidth=1.8, label=labels[v])
    axes[0].axhline(eps, color="k", linestyle=":", linewidth=1.4,
                    label=f"$\\epsilon = {eps:.3f}$")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Training round")
    axes[0].set_ylabel("Accuracy error (log scale)")
    axes[0].set_title("Theoretical (solid) vs empirical (dashed) per-round error")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_xlabel("Training round")
    axes[1].set_ylabel(r"Mean $\kappa_i$ EMA (empirical / theoretical)")
    axes[1].set_title("EF recovery factor over training")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig10_bound_tightness.png"), dpi=140)
    plt.close(fig)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiers", type=str, default="2tier",
                    help="Match the --tiers used in run_pareto.py (2tier/3tier/4tier/custom)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    pareto_rounds = _load_csv(os.path.join(args.out, "pareto_rounds.csv"))
    summary = _load_csv(os.path.join(args.out, "summary.csv"))
    adaptive = _load_csv(os.path.join(args.out, "adaptive_rounds.csv"))

    # Get bandwidth array for fig 2. Must match the --tiers used in run_pareto.
    from compressiq.cluster import make_heterogeneous_profiles, make_tiered_profiles
    tier_presets = {
        "2tier": None,  # use make_heterogeneous_profiles for exact reproducibility
        "3tier": [(10.0, 5.0, 0.5), (5.0, 20.0, 0.25), (1.0, 50.0, 0.25)],
        "4tier": [(10.0, 5.0, 0.375), (5.0, 15.0, 0.25),
                  (2.5, 30.0, 0.25), (1.0, 50.0, 0.125)],
    }
    if args.tiers == "2tier":
        profiles = make_heterogeneous_profiles(args.n_workers, seed=args.seed)
    elif args.tiers in tier_presets:
        profiles = make_tiered_profiles(args.n_workers, tier_presets[args.tiers], seed=args.seed)
    else:
        tiers = [tuple(float(v) for v in part.split(":")) for part in args.tiers.split(",")]
        profiles = make_tiered_profiles(args.n_workers, tiers, seed=args.seed)
    bws_gbps = np.array([p.bandwidth_bps / 1e9 for p in profiles])

    if pareto_rounds is not None:
        fig1_pareto(pareto_rounds, args.out); print("wrote fig1_pareto.png")
        fig3_round_time(pareto_rounds, args.out); print("wrote fig3_round_time.png")
        fig8_eps_sensitivity(pareto_rounds, args.out); print("wrote fig8_eps_sensitivity.png")
    if summary is not None:
        fig2_ratios(summary, bws_gbps, args.out); print("wrote fig2_ratios.png")
    fig4_model_validation(args.out, n_workers=args.n_workers, seed=args.seed)
    print("wrote fig4_model_validation.png")
    if adaptive is not None:
        fig5_constraint(adaptive, args.out); print("wrote fig5_constraint.png")
        fig6_alpha_drift(adaptive, args.out); print("wrote fig6_alpha_drift.png")
    if summary is not None and adaptive is not None:
        fig7_speedup(summary, adaptive, args.out); print("wrote fig7_speedup.png")

    refine = _load_csv(os.path.join(args.out, "refine_rounds.csv"))
    refine_summary = _load_csv(os.path.join(args.out, "refine_summary.csv"))
    if refine is not None and refine_summary is not None:
        fig9_refine_speed(refine, refine_summary, args.out); print("wrote fig9_refine_speed.png")
        fig10_bound_tightness(refine, args.out); print("wrote fig10_bound_tightness.png")


if __name__ == "__main__":
    main()
