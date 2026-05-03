# CompressIQ

**Bandwidth-aware per-worker gradient compression for heterogeneous distributed training, formulated as a convex program.**

Synchronous data-parallel training on heterogeneous clusters (mixed-bandwidth links, multi-AZ cloud, on-prem + cloud) is bottlenecked by the slowest worker. Uniform gradient compression — the standard production practice — throttles fast workers to match slow ones, wasting the faster links. CompressIQ solves a small convex program to pick a *different* compression ratio for each worker, keeping them all on-schedule with the ring all-reduce while staying under a global accuracy-loss budget.

On a 12-worker 3-tier cluster (10 / 5 / 1 Gbps), CompressIQ delivers a **1.85× speedup** over uniform compression at matched test accuracy. Two optional bound refinements — an **EF-aware** online correction and a **per-layer** extension — add a further **1.76×**, for a cumulative **3.26× improvement**.

---

## Key results

| Scheme | Final test acc (MNIST) | Total sim comm time | Speedup vs uniform |
|---|---|---|---|
| No compression | 0.9072 | baseline × 4.2 | 0.24× |
| Uniform compression | 0.9058 | 1.00× | 1.00× |
| Greedy per-worker | 0.9049 | 0.71× | 1.41× |
| **CompressIQ (static)** | **0.9055** | **0.54×** | **1.85×** |
| CompressIQ + adaptive recalibration | 0.9063 | 0.54× | 1.85× |
| CompressIQ + EF-aware bound | 0.9061 | 0.34× | 2.91× |
| CompressIQ + per-layer | 0.9060 | 0.47× | 2.15× |
| **CompressIQ + per-layer + EF-aware** | **0.9071** | **0.31×** | **3.26×** |

Figures in `results/` reproduce the full Pareto frontier, per-worker ratio water-filling, bound validation, α-drift dynamics, and refinement ablations.

---

## What's in the optimizer

**The core problem.** Given `N` heterogeneous workers with outbound link bandwidths `B_i` (bits/s) and per-hop latencies `α_i`, a gradient of size `G` bits, per-worker accuracy weights `α_i = ‖g_i‖²`, and an accuracy budget `ε`, find the per-worker compression ratios `r_i ∈ [r_min, 1]` that minimize the ring all-reduce bottleneck time:

```
minimize    max_i   r_i / B_i
subject to  Σ_i  α_i · (1 − r_i)²   ≤  ε
            r_min  ≤  r_i  ≤  1       ∀i
```

- The objective is a `max` of linear functions (convex).
- The accuracy constraint is a sum of convex quadratics.
- Solved with CVXPY (Clarabel / ECOS / SCS); typical solve time < 10 ms for `N = 12`.

**EF-aware refinement** (Karimireddy et al. 2019). The naive bound `(1 − r)² · α` is a single-round upper bound; under error feedback, past residuals get re-transmitted so the *actual* leaked error is 20-30× smaller in steady state. We measure the per-worker correction `κ_i = EMA(empirical / theoretical)` online and rescale the constraint:

```
Σ_i  κ_i · α_i · (1 − r_i)²   ≤   ε
```

This lets the optimizer correctly identify that it has ~30× more usable budget than the naive bound suggests.

**Per-layer refinement.** The scalar `r_i` conflates layers of very different sizes. Replacing it with a matrix `R[i, ℓ]` (worker i, layer ℓ) keeps the problem convex:

```
minimize    max_i   (1/B_i) Σ_ℓ R[i,ℓ] · G_ℓ
subject to  Σ_{i,ℓ}  α_{i,ℓ} · (1 − R[i,ℓ])²   ≤  ε
```

The solver automatically learns to keep small, accuracy-sensitive layers (e.g. the output head) near-uncompressed while aggressively compressing the large hidden layers.

---

## Architecture & cost model

The simulator advances simulated wall-clock time by the ring all-reduce round time of Patarasuk & Yuan (2009):

```
T_round = (2(N-1)/N) · G · max_i (r_i / B_i)  +  2(N-1) · max_i α_i
         \__________________________________/   \__________________/
                  bandwidth term                    latency term
```

Each `GPUWorker` owns an MLP replica (784-128-128-10, 118 K parameters), an IID MNIST shard, and a `NetworkProfile(bandwidth, latency)`. A training round is: local forward+backward → Top-K compress (with error feedback) → simulated ring all-reduce → apply averaged gradient.

See `results/fig0_architecture.png` for the full experimental setup diagram.

---

## Repository layout

```
compressiq/
  cost_model.py     bandwidth-optimal ring all-reduce timing
  compression.py    Top-K sparsification + error feedback
  optimizer.py      CVXPY solver + baselines (none/uniform/greedy) + EF-aware + per-layer
  worker.py         GPUWorker abstraction (MLP + shard + network profile)
  cluster.py        cluster construction (2-tier / 3-tier / 4-tier heterogeneity)
  simulator.py      discrete-event training loop with adaptive recalibration
  data.py           MNIST loading + IID sharding

experiments/
  run_pareto.py                 Pareto frontier sweep (scheme × epsilon)
  run_adaptive.py               adaptive recalibration vs static comparison
  run_sensitivity.py            round time vs cluster size N
  run_bound_refinements.py      4-way ablation (naive / EF-aware / per-layer / both)
  make_plots.py                 regenerate all 10 figures from CSVs
  make_architecture_diagram.py  fig0 experimental setup diagram
  make_report.py                build results/report.docx

results/                        CSVs + figures + report.docx (small; tracked)
```

---

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. Reproduce the full Pareto frontier (~5 min)
python experiments/run_pareto.py --n-workers 12 --rounds 200 --tiers 3tier

# 2. Adaptive recalibration experiment
python experiments/run_adaptive.py --n-workers 12 --rounds 300 --tiers 3tier

# 3. Bound refinements (EF-aware + per-layer)
python experiments/run_bound_refinements.py --n-workers 12 --rounds 200

# 4. Regenerate all plots from the CSVs
python experiments/make_plots.py --n-workers 12 --tiers 3tier

# 5. Rebuild the report
python experiments/make_report.py
```

All outputs land in `results/`.

---

## Figures in `results/`

| File | Description |
|---|---|
| `fig0_architecture.png` | 12-worker 3-tier ring-all-reduce architecture diagram |
| `fig1_pareto.png` | Pareto frontier: accuracy vs comm time across all schemes |
| `fig2_ratios.png` | Per-worker compression ratios; water-filling structure |
| `fig3_round_time.png` | Round-time decomposition (bandwidth + latency) |
| `fig4_model_validation.png` | Top-K empirical error vs naive `(1−r)² · α` bound |
| `fig5_constraint.png` | Theoretical vs empirical per-round error over training |
| `fig6_alpha_drift.png` | Per-worker `α_i = ‖g_i‖²` drift during training |
| `fig7_speedup.png` | Headline speedup bars: CompressIQ vs baselines |
| `fig8_eps_sensitivity.png` | Sensitivity of comm time to accuracy budget `ε` |
| `fig9_refine_speed.png` | Bound-refinement ablation: accuracy-vs-time + speedup bars |
| `fig10_bound_tightness.png` | Theoretical vs empirical error for each refinement; `κ_i` EMA |

---

## References

- Patarasuk & Yuan, *Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations*, JPDC 2009.
- Karimireddy, Rebjock, Stich, Jaggi, *Error Feedback Fixes SignSGD and other Gradient Compression Schemes*, ICML 2019.
- Lin et al., *Deep Gradient Compression*, ICLR 2018.
- Alistarh et al., *QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding*, NeurIPS 2017.
- Vogels, Karimireddy, Jaggi, *PowerSGD*, NeurIPS 2019.
- Diamond & Boyd, *CVXPY: A Python-Embedded Modeling Language for Convex Optimization*, JMLR 2016.
- Sergeev & Del Balso, *Horovod*, 2018.

---

## License

MIT (see [LICENSE](LICENSE) if present; otherwise repo is provided for academic review).

