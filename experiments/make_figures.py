"""
Phase-4 figure generation for the paper's Section 4 (Results).

Reads experiments/full_runs/summary/*.csv (produced by aggregate_results.py)
and writes print-ready PDF figures into ../latex_source/figures/. Uses the
validated 8-hue categorical palette (colorblind-safe, fixed slot order) from
the project's dataviz guidance: slot 1 blue / slot 2 orange are used for the
two-series comparisons; the 8-way ablation panels are grayscale-single-hue
small multiples (identity is carried by panel position/label, not color, so
no palette collision across 8 series).

Outputs:
  fig_learning_curves.pdf   - 8-panel small multiples, reward vs. episode,
                              mean +/- std band across 3 seeds, one panel per
                              ablation config.
  fig_ablation_bars.pdf     - bar chart of final-window mean reward +/- std
                              per config, baseline (D3QN+PER) highlighted.
  fig_sensing_sweep.pdf     - 4-panel small multiples (beam count / FOV /
                              noise / dropout), mean checkpoints vs. level,
                              nominal- vs randomized-trained regime.

Usage:
    python experiments/make_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUMMARY_DIR = "experiments/full_runs/summary"
FIG_DIR = "../latex_source/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Validated categorical palette (fixed order, colorblind-safe) — see
# dataviz skill references/palette.md. Slot 1 = blue, slot 2 = orange.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#8a8a86"

CONFIG_LABELS = {
    "dqn": "DQN", "dqn_per": "DQN+PER",
    "ddqn": "Double DQN", "ddqn_per": "Double DQN+PER",
    "dueling": "Dueling DQN", "dueling_per": "Dueling DQN+PER",
    "d3qn": "D3QN (no PER)", "d3qn_per": "D3QN+PER (ours)",
}
CONFIG_ORDER = ["dqn", "dqn_per", "ddqn", "ddqn_per",
                "dueling", "dueling_per", "d3qn", "d3qn_per"]

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

# The paper places every figure at \textwidth or \columnwidth, which in the
# current (onecolumn) IEEEtran layout is 516pt = 516/72.27in = 7.14in. Each
# figure's native figsize width differs, so matplotlib's default font.size=9
# renders at a different EFFECTIVE size once LaTeX scales the PDF down/up to
# that placement width (e.g. an 11in-wide figure at 9pt shrinks to ~5.8pt on
# the page). To keep on-page text visually consistent with the ~8.5pt of the
# document's captions/table text, each figure's base font.size is set so
# that (base_font * placement_width / figure_width) ~= 8.5pt.
PLACEMENT_WIDTH_IN = 516.0 / 72.27
TARGET_PT = 8.5


def _base_font_for(fig_width_in):
    return TARGET_PT * fig_width_in / PLACEMENT_WIDTH_IN


def fig_learning_curves():
    df = pd.read_csv(os.path.join(SUMMARY_DIR, "ablation_learning_curves.csv"))
    fig_w = 11
    base = _base_font_for(fig_w)
    with plt.rc_context({"font.size": base}):
        fig, axes = plt.subplots(2, 4, figsize=(fig_w, 5), sharex=True, sharey=True)
        for ax, cfg in zip(axes.flat, CONFIG_ORDER):
            sub = df[df["config"] == cfg]
            if sub.empty:
                ax.set_visible(False)
                continue
            grouped = sub.groupby("episode")["total_reward"]
            mean = grouped.mean()
            std = grouped.std().fillna(0.0)
            color = ORANGE if cfg == "d3qn_per" else BLUE
            ax.plot(mean.index, mean.values, color=color, linewidth=1.4)
            ax.fill_between(mean.index, mean.values - std.values, mean.values + std.values,
                             color=color, alpha=0.2, linewidth=0)
            ax.set_title(CONFIG_LABELS[cfg], fontsize=base,
                         fontweight="bold" if cfg == "d3qn_per" else "normal")
            ax.axhline(0, color=GRAY, linewidth=0.5, linestyle="--")
        for ax in axes[-1, :]:
            ax.set_xlabel("Episode")
        for ax in axes[:, 0]:
            ax.set_ylabel("Episode reward")
        fig.suptitle("Training learning curves across the component ablation matrix "
                      "(mean $\\pm$ 1 std over 3 seeds)", fontsize=base * 10 / 9)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(FIG_DIR, "fig_learning_curves.pdf"))
        plt.close(fig)


def fig_ablation_bars():
    df = pd.read_csv(os.path.join(SUMMARY_DIR, "ablation_summary.csv"))
    df["config"] = pd.Categorical(df["config"], categories=CONFIG_ORDER, ordered=True)
    df = df.sort_values("config")
    labels = [CONFIG_LABELS[c] for c in df["config"]]
    colors = [ORANGE if c == "d3qn_per" else BLUE for c in df["config"]]

    fig_w = 7
    base = _base_font_for(fig_w)
    with plt.rc_context({"font.size": base}):
        fig, ax = plt.subplots(figsize=(fig_w, 4))
        x = np.arange(len(df))
        ax.bar(x, df["mean_reward_final"], yerr=df["std_reward_final"], color=colors,
               capsize=3, width=0.65, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Final-window mean episode reward\n(last 20 of 120 episodes)")
        ax.set_title("Ablation matrix: final training performance by architectural component")
        for xi, row in zip(x, df.itertuples()):
            marker = "" if row.config == "d3qn_per" else (
                "*" if getattr(row, "welch_p_vs_baseline") < 0.05 else "")
            if marker:
                ax.text(xi, row.mean_reward_final + row.std_reward_final + 10, marker,
                        ha="center", va="bottom", fontsize=base * 11 / 9, color=GRAY)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig_ablation_bars.pdf"))
        plt.close(fig)


def fig_sensing_sweep():
    df = pd.read_csv(os.path.join(SUMMARY_DIR, "sensing_summary.csv"))
    factors = [
        ("beam_count", "Beam count", None),
        ("fov_deg", "Field of view (deg)", None),
        ("noise_sigma", "Range noise $\\sigma$", None),
        ("dropout_p", "Per-beam dropout $p$", None),
    ]
    fig_w = 13
    base = _base_font_for(fig_w)
    with plt.rc_context({"font.size": base}):
        fig, axes = plt.subplots(1, 4, figsize=(fig_w, 3.4), sharey=True)
        for ax, (factor, xlabel, _) in zip(axes, factors):
            sub = df[df["factor"] == factor].sort_values("level")
            for regime, color, marker in [("nominal", BLUE, "o"), ("randomized", ORANGE, "s")]:
                r = sub[sub["regime"] == regime]
                ax.errorbar(r["level"], r["mean_checkpoints"], yerr=r["std_checkpoints"],
                            color=color, marker=marker, markersize=4, linewidth=1.4,
                            capsize=2, label="Nominal-trained" if regime == "nominal" else "Randomized-trained")
            ax.set_xlabel(xlabel)
        axes[0].set_ylabel("Mean checkpoints cleared\n(of 12, per episode)")
        axes[0].legend(loc="upper right", fontsize=base * 7 / 9, frameon=False)
        fig.suptitle("Sensing-degradation sweep: evaluated robustness (nominal-trained) vs. "
                     "trained robustness (randomized-trained)", fontsize=base * 10 / 9)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(os.path.join(FIG_DIR, "fig_sensing_sweep.pdf"))
        plt.close(fig)


def fig_advantage_dispersion():
    path = os.path.join(SUMMARY_DIR, "advantage_dispersion_summary.csv")
    if not os.path.exists(path):
        print("no advantage_dispersion_summary.csv, skipping fig_advantage_dispersion")
        return
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["mean_advantage_std", "mean_checkpoints", "regime"])
    fig_w = 5.5
    base = _base_font_for(fig_w)
    with plt.rc_context({"font.size": base}):
        fig, ax = plt.subplots(figsize=(fig_w, 4.2))
        for regime, color, marker in [("nominal", BLUE, "o"), ("randomized", ORANGE, "s")]:
            sub = df[df["regime"] == regime]
            ax.scatter(sub["mean_checkpoints"], sub["mean_advantage_std"], color=color,
                       marker=marker, s=36, label="Nominal-trained" if regime == "nominal" else "Randomized-trained",
                       zorder=3)
            if len(sub) > 1:
                coeffs = np.polyfit(sub["mean_checkpoints"], sub["mean_advantage_std"], 1)
                xs = np.linspace(sub["mean_checkpoints"].min(), sub["mean_checkpoints"].max(), 20)
                ax.plot(xs, np.polyval(coeffs, xs), color=color, linewidth=1.0, alpha=0.5, zorder=2)
        ax.set_xlabel("Mean checkpoints cleared (of 12)")
        ax.set_ylabel("Mean advantage-stream dispersion\n(std across 4 actions)")
        ax.set_title("Advantage dispersion vs. navigation competence\nacross the sensing sweep")
        ax.legend(loc="upper right", fontsize=base * 8 / 9, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig_advantage_dispersion.pdf"))
        plt.close(fig)




if __name__ == "__main__":
    fig_learning_curves()
    fig_ablation_bars()
    fig_sensing_sweep()
    fig_advantage_dispersion()
    print(f"Wrote figures to {FIG_DIR}/")
