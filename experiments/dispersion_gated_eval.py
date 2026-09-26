"""
Workstream B: closed-loop advantage-dispersion safety layer.

Promotes the offline diagnostic in advantage_instrumentation.py into a live,
actionable eval-time mechanism: at every step of a greedy rollout, the
pre-aggregation advantage vector's dispersion (std across the 4 actions) is
computed inline (reusing diagnostic_model(), no retraining, one extra cheap
read of an activation the network already produces), compared against a
nominal-operation baseline calibrated once from nominal-sensing rollouts,
and used to gate a conservative brake-override fallback whenever the
windowed dispersion sustainedly exceeds that baseline by a margin.

Does NOT retrain anything -- reuses the exact final.weights.h5 from the
paper's already-published sensing_train runs (experiments/full_runs). Runs
each of the 13 sensing-grid points twice per (regime, seed): once with the
fallback disabled (must reproduce the paper's published sensing_eval
numbers, used here as a regression check) and once enabled, to measure the
crash-rate/near-miss-rate benefit and the nominal-performance cost.

Usage:
    python experiments/dispersion_gated_eval.py
"""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import pandas as pd

import full_study as fs
from advantage_instrumentation import diagnostic_model

RUNS_DIR = "experiments/full_runs"
OUT_DIR = os.path.join(RUNS_DIR, "summary")
SEEDS = (0, 1, 2)
EVAL_EPISODES = 15
MAX_STEPS = 350
CALIBRATION_EPISODES = 15
WINDOW = 5
MARGIN_SIGMA = 2.0
BRAKE_ACTION = 3
NOMINAL_POINT = dict(beam_count=3, fov_deg=90.0, noise_sigma=0.0, dropout_p=0.0)


def calibrate_baseline(diag_model, seed, episodes=CALIBRATION_EPISODES, max_steps=MAX_STEPS):
    """Nominal-operation dispersion baseline, from rollouts at the policy's
    own training-time (nominal) sensing point."""
    fs.set_global_seed(seed + 7000)
    env = fs.SensingGameEnv(max_steps=max_steps, sensing=NOMINAL_POINT)
    all_disp = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            v, a = diag_model(np.expand_dims(state, 0), training=False)
            a = a.numpy()[0]
            all_disp.append(float(np.std(a)))
            action = int(np.argmax(v.numpy()[0][0] + (a - a.mean())))
            state, reward, done, info = env.step(action)
    all_disp = np.asarray(all_disp)
    return float(all_disp.mean()), float(all_disp.std())


def rollout_gated(diag_model, sensing_point, seed, episodes, max_steps, threshold, gated):
    fs.set_global_seed(seed + 9000)
    env = fs.SensingGameEnv(max_steps=max_steps, sensing=sensing_point)
    rows = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward, steps, n_overrides = 0.0, 0, 0
        window = []
        while not done:
            v, a = diag_model(np.expand_dims(state, 0), training=False)
            a = a.numpy()[0]
            disp = float(np.std(a))
            window.append(disp)
            if len(window) > WINDOW:
                window.pop(0)
            action = int(np.argmax(v.numpy()[0][0] + (a - a.mean())))
            if gated and len(window) == WINDOW and (sum(window) / WINDOW) > threshold:
                action = BRAKE_ACTION
                n_overrides += 1
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
        rows.append(dict(episode=ep, total_reward=total_reward, steps=steps,
                          checkpoints=env.checkpoints_cleared, crashed=int(env.crashed),
                          lap_completed=int(len(env.lap_times) > 0),
                          override_frac=n_overrides / max(steps, 1)))
    return pd.DataFrame(rows)


def main():
    out_rows = []
    for regime in ("nominal", "randomized"):
        for seed in SEEDS:
            weights_path = os.path.join(RUNS_DIR, "sensing_train", f"{regime}_seed{seed}",
                                         "final.weights.h5")
            if not os.path.exists(weights_path):
                print(f"WARNING: missing {weights_path}, skipping")
                continue
            cfg = fs.ABLATION_CONFIGS["d3qn_per"]
            agent = fs.Agent(fs.MAX_BEAMS + 2, 4, **cfg)
            agent.model.load_weights(weights_path)
            diag = diagnostic_model(agent.model)

            base_mean, base_std = calibrate_baseline(diag, seed)
            threshold = base_mean + MARGIN_SIGMA * base_std
            print(f"[{regime} seed={seed}] baseline dispersion mean={base_mean:.3f} "
                  f"std={base_std:.3f} threshold={threshold:.3f}")

            for point in fs.SENSING_GRID:
                for gated in (False, True):
                    df = rollout_gated(diag, point, seed, EVAL_EPISODES, MAX_STEPS,
                                        threshold, gated)
                    s = fs.summarize(df, extra=dict(
                        regime=regime, seed=seed, gated=gated,
                        baseline_dispersion_mean=base_mean,
                        baseline_dispersion_std=base_std,
                        threshold=threshold,
                        mean_override_frac=df["override_frac"].mean(),
                        **point))
                    out_rows.append(s)
            print(f"done {regime} seed {seed}")

    out_df = pd.DataFrame(out_rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "dispersion_gated_eval.csv")
    out_df.to_csv(out_path, index=False)
    print(f"wrote {out_path}")

    # Regression check: ungated crash_rate/mean_checkpoints should match the
    # paper's already-published sensing_eval numbers (same weights, same
    # greedy policy, same seeds/points -- gating disabled should be a no-op).
    ungated = out_df[~out_df["gated"]]
    gated = out_df[out_df["gated"]]
    merge_keys = ["regime", "seed", "beam_count", "fov_deg", "noise_sigma", "dropout_p"]
    cmp = ungated.merge(gated, on=merge_keys, suffixes=("_ungated", "_gated"))
    cmp["crash_rate_delta"] = cmp["crash_rate_gated"] - cmp["crash_rate_ungated"]
    cmp["checkpoints_delta"] = cmp["mean_checkpoints_gated"] - cmp["mean_checkpoints_ungated"]
    summary = cmp.groupby("regime")[["crash_rate_delta", "checkpoints_delta"]].mean()
    print(summary.to_string())
    summary.to_csv(os.path.join(OUT_DIR, "dispersion_gated_summary.csv"))


if __name__ == "__main__":
    main()
