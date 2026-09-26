"""
Phase-4 results aggregation for the paper's Section 4 (Results).

Reads the raw output of `full_study.py --phase driver` (already run into
experiments/full_runs/) and produces the summary tables the paper's
Section 3 "Benchmarking Protocol" already promises:

  1. Ablation matrix summary: for each of the 8 configs (DQN/Double/Dueling/
     D3QN x {no PER, PER}), across 3 seeds, the final-window training
     performance (mean reward, mean checkpoints cleared, crash rate over the
     last 20 of 120 episodes) plus a significance test (Welch's t-test and
     Mann-Whitney U, since n=3 per arm is too small to assume normality
     safely either way) and an effect size (Cohen's d) against the full
     D3QN+PER baseline.
  2. Sensing-sweep summary: for each of the 4 swept factors (beam count,
     FOV, noise, dropout), the evaluated-robustness (nominal-trained) vs.
     trained-robustness (randomized-trained) checkpoint-clearance mean +/-
     std across 3 seeds at each grid point, one row per point.

Outputs (all under experiments/full_runs/summary/):
  - ablation_summary.csv       (one row per config, ready for the paper's
                                 results table)
  - ablation_learning_curves.csv (per-config, per-episode mean+std reward
                                 across seeds, for the learning-curve figure)
  - sensing_summary.csv        (one row per (factor, level, regime), ready
                                 for the sensing-degradation figure)

Usage:
    python experiments/aggregate_results.py
"""
import os
import json
import numpy as np
import pandas as pd
from scipy import stats

RUNS_DIR = "experiments/full_runs"
OUT_DIR = os.path.join(RUNS_DIR, "summary")
os.makedirs(OUT_DIR, exist_ok=True)

ABLATION_CONFIGS = [
    "dqn", "dqn_per", "ddqn", "ddqn_per",
    "dueling", "dueling_per", "d3qn", "d3qn_per",
]
BASELINE = "d3qn_per"
SEEDS = (0, 1, 2)
FINAL_WINDOW = 20  # last N of 120 episodes, matches the phase-switch window
                    # already used elsewhere in the training protocol (Sec. 3)


def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_std = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


# --------------------------------------------------------------------- #
# 1. Ablation matrix
# --------------------------------------------------------------------- #
def load_ablation():
    per_seed_final = {}   # config -> list of per-seed final-window mean reward
    per_seed_metrics = {} # config -> list of dicts (reward, checkpoints, crash_rate, best_checkpoints)
    curves = []            # rows for learning-curve CSV

    for cfg in ABLATION_CONFIGS:
        per_seed_final[cfg] = []
        per_seed_metrics[cfg] = []
        for seed in SEEDS:
            path = os.path.join(RUNS_DIR, "ablation", f"{cfg}_seed{seed}", "train_log.csv")
            if not os.path.exists(path):
                print(f"WARNING: missing {path}, skipping")
                continue
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                curves.append(dict(config=cfg, seed=seed, episode=int(row["episode"]),
                                    total_reward=row["total_reward"], checkpoints=row["checkpoints"]))
            tail = df.tail(FINAL_WINDOW)
            per_seed_final[cfg].append(tail["total_reward"].mean())
            per_seed_metrics[cfg].append(dict(
                seed=seed,
                mean_reward_final=tail["total_reward"].mean(),
                mean_checkpoints_final=tail["checkpoints"].mean(),
                crash_rate_final=tail["crashed"].mean(),
                best_checkpoints=df["checkpoints"].max(),
                max_reward=df["total_reward"].max(),
            ))
    return per_seed_final, per_seed_metrics, pd.DataFrame(curves)


def summarize_ablation(per_seed_final, per_seed_metrics):
    baseline_vals = per_seed_final[BASELINE]
    rows = []
    for cfg in ABLATION_CONFIGS:
        vals = per_seed_final[cfg]
        metrics = per_seed_metrics[cfg]
        if len(vals) == 0:
            continue
        mean_r = np.mean(vals)
        std_r = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
        mean_cp = np.mean([m["mean_checkpoints_final"] for m in metrics])
        std_cp = np.std([m["mean_checkpoints_final"] for m in metrics], ddof=1) if len(metrics) > 1 else 0.0
        mean_crash = np.mean([m["crash_rate_final"] for m in metrics])
        best_cp = max(m["best_checkpoints"] for m in metrics)

        if cfg == BASELINE:
            p_welch, p_mw, d = float("nan"), float("nan"), 0.0
        else:
            try:
                p_welch = stats.ttest_ind(vals, baseline_vals, equal_var=False).pvalue
            except Exception:
                p_welch = float("nan")
            try:
                p_mw = stats.mannwhitneyu(vals, baseline_vals, alternative="two-sided").pvalue
            except Exception:
                p_mw = float("nan")
            d = cohens_d(vals, baseline_vals)

        rows.append(dict(
            config=cfg, n_seeds=len(vals),
            mean_reward_final=mean_r, std_reward_final=std_r,
            mean_checkpoints_final=mean_cp, std_checkpoints_final=std_cp,
            mean_crash_rate_final=mean_crash,
            best_checkpoints_overall=best_cp,
            welch_p_vs_baseline=p_welch, mannwhitney_p_vs_baseline=p_mw,
            cohens_d_vs_baseline=d,
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# 2. Sensing sweep
# --------------------------------------------------------------------- #
FACTORS = {
    "beam_count": (1, 3, 5, 7),
    "fov_deg": (60, 90, 120, 150),
    "noise_sigma": (0.0, 0.02, 0.05, 0.1),
    "dropout_p": (0.0, 0.1, 0.2, 0.4),
}
NOMINAL_POINT = dict(beam_count=3, fov_deg=90.0, noise_sigma=0.0, dropout_p=0.0)


def load_sensing():
    frames = []
    for regime in ("nominal", "randomized"):
        for seed in SEEDS:
            path = os.path.join(RUNS_DIR, "sensing_eval", f"{regime}_seed{seed}.csv")
            if not os.path.exists(path):
                print(f"WARNING: missing {path}, skipping")
                continue
            df = pd.read_csv(path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_sensing(df):
    rows = []
    for factor, levels in FACTORS.items():
        other_factors = {k: v for k, v in NOMINAL_POINT.items() if k != factor}
        for level in levels:
            mask = (df[factor] == level)
            for k, v in other_factors.items():
                mask &= np.isclose(df[k].astype(float), float(v))
            for regime in ("nominal", "randomized"):
                sub = df[mask & (df["regime"] == regime)]
                if len(sub) == 0:
                    continue
                rows.append(dict(
                    factor=factor, level=level, regime=regime, n_seeds=len(sub),
                    mean_checkpoints=sub["mean_checkpoints"].mean(),
                    std_checkpoints=sub["mean_checkpoints"].std(ddof=1) if len(sub) > 1 else 0.0,
                    mean_reward=sub["mean_reward"].mean(),
                    crash_rate=sub["crash_rate"].mean(),
                    checkpoint_clear_rate=sub["checkpoint_clear_rate"].mean(),
                ))
    return pd.DataFrame(rows)


def main():
    print("Loading ablation matrix...")
    per_seed_final, per_seed_metrics, curves_df = load_ablation()
    ablation_summary = summarize_ablation(per_seed_final, per_seed_metrics)
    ablation_summary.to_csv(os.path.join(OUT_DIR, "ablation_summary.csv"), index=False)
    curves_df.to_csv(os.path.join(OUT_DIR, "ablation_learning_curves.csv"), index=False)
    print(ablation_summary.to_string(index=False))

    print("\nLoading sensing sweep...")
    sensing_df = load_sensing()
    sensing_summary = summarize_sensing(sensing_df)
    sensing_summary.to_csv(os.path.join(OUT_DIR, "sensing_summary.csv"), index=False)
    print(sensing_summary.to_string(index=False))

    print(f"\nWrote summaries to {OUT_DIR}/")


if __name__ == "__main__":
    main()
