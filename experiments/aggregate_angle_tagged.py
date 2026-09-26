"""
Workstream A: three-way encoding comparison.

Aggregates the angle-tagged sensing sweep (experiments/full_runs_angle_tagged/
sensing_eval/) the same way aggregate_results.py aggregates the paper's
already-published ordinal-encoding baseline (experiments/full_runs/
sensing_eval/), then merges both into one comparison table answering the
question Workstream A was designed to answer: does fixing the beam-count
representation (angle-tagged encoding) close the beam-count brittleness gap
that randomized-sensing training only partially closed, on top of the
ordinal-encoding baseline -- and does combining both close it further?

Usage:
    python experiments/aggregate_angle_tagged.py
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_results import summarize_sensing, FACTORS  # noqa: E402

BASELINE_RUNS_DIR = "experiments/full_runs"
ANGLE_RUNS_DIR = "experiments/full_runs_angle_tagged"
OUT_DIR = os.path.join(BASELINE_RUNS_DIR, "summary")
SEEDS = (0, 1, 2)


def load_sensing(runs_dir):
    frames = []
    for regime in ("nominal", "randomized"):
        for seed in SEEDS:
            path = os.path.join(runs_dir, "sensing_eval", f"{regime}_seed{seed}.csv")
            if not os.path.exists(path):
                print(f"WARNING: missing {path}, skipping")
                continue
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def main():
    baseline_df = load_sensing(BASELINE_RUNS_DIR)
    angle_df = load_sensing(ANGLE_RUNS_DIR)

    baseline_summary = summarize_sensing(baseline_df)
    baseline_summary["encoding"] = "ordinal"
    angle_summary = summarize_sensing(angle_df)
    angle_summary["encoding"] = "angle_tagged"

    combined = pd.concat([baseline_summary, angle_summary], ignore_index=True)
    combined = combined[["encoding", "regime", "factor", "level", "n_seeds",
                          "mean_checkpoints", "std_checkpoints", "mean_reward",
                          "crash_rate", "checkpoint_clear_rate"]]
    out_path = os.path.join(OUT_DIR, "encoding_comparison_summary.csv")
    combined.to_csv(out_path, index=False)
    print(f"wrote {out_path}\n")

    print("=== Beam-count factor: the whole point of Workstream A ===")
    bc = combined[combined["factor"] == "beam_count"].sort_values(
        ["level", "encoding", "regime"])
    print(bc.to_string(index=False))

    print("\n=== Reference point (level=3/90deg nominal beam config) across all four arms ===")
    ref = combined[(combined["factor"] == "beam_count") & (combined["level"] == 3)]
    print(ref.to_string(index=False))


if __name__ == "__main__":
    main()
